"""Adversarial training loop for concept hiding.

Implements the minimax objective:
    min_θ [ L_task(θ) + α·L_anchor(θ,θ_ref) - λ·L_probe(φ*, h_L(θ)) ]

Inner loop: train linear probe on detached activations (SGD, lr=3e-3).
Outer loop: update model to maintain task performance + maximize probe loss.
"""

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.activation_extractor import extract_activations, extract_all_activations
from src.probes.probe import LinearProbe, SklearnProbe, train_linear_probe_inner

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    # Outer loop
    outer_lr: float = 2e-5
    outer_weight_decay: float = 0.01
    outer_steps: int = 5000
    task_batch_size: int = 16
    max_grad_norm: float = 1.0
    alpha: float = 0.1  # anchor (KL) weight
    lambda_max: float = 1.0  # max probe adversarial weight
    lambda_warmup_steps: int = 1000

    # Inner loop
    inner_lr: float = 3e-3
    inner_weight_decay: float = 1e-4
    inner_steps: int = 5
    probe_batch_size: int = 256
    probe_reinit_every: int = 200

    # Stability
    sft_every: int = 50  # pure SFT step every N outer steps
    checkpoint_every: int = 100
    eval_every: int = 100  # probe eval + generation logging
    early_stop_window: int = 500
    early_stop_auc_threshold: float = 0.55
    early_stop_ppl_threshold: float = 1.20  # within 20% of baseline

    # Model
    target_layer: int = 15  # set from Phase 0 sweep
    hidden_dim: int = 1024
    max_seq_len: int = 512
    device: str = "cuda"
    dtype: str = "bfloat16"

    # Generation logging
    gen_prompts: list[str] = field(default_factory=lambda: [
        "The movie was absolutely",
        "I hated every minute of",
        "This is a wonderful",
        "The food at this restaurant was",
        "I feel so disappointed because",
        "The weather today is beautiful and",
    ])
    gen_max_new_tokens: int = 50

    # Logging
    checkpoint_dir: str = "checkpoints"
    use_wandb: bool = True


class AdversarialTrainer:
    def __init__(
        self,
        model: nn.Module,
        task_loader: DataLoader,
        probe_loader: DataLoader,
        anchor_loader: DataLoader,
        config: TrainConfig,
        tokenizer=None,
        eval_loader: DataLoader | None = None,
        eval_labels: np.ndarray | None = None,
    ):
        self.config = config
        self.device = config.device
        self.dtype = getattr(torch, config.dtype)
        self.tokenizer = tokenizer

        # Eval data for periodic probe evaluation
        self.eval_loader = eval_loader
        self.eval_labels = eval_labels

        # Model and frozen reference
        self.model = model.to(self.device)
        self.ref_model = copy.deepcopy(model).to(self.device)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

        # Data loaders
        self.task_loader = task_loader
        self.probe_loader = probe_loader
        self.anchor_loader = anchor_loader
        self.task_iter = iter(self.task_loader)
        self.probe_iter = iter(self.probe_loader)
        self.anchor_iter = iter(self.anchor_loader)

        # Probe (linear, differentiable)
        self.probe = LinearProbe(config.hidden_dim).to(self.device)
        self.probe_optimizer = torch.optim.SGD(
            self.probe.parameters(),
            lr=config.inner_lr,
            weight_decay=config.inner_weight_decay,
        )

        # Model optimizer
        self.model_optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.outer_lr,
            weight_decay=config.outer_weight_decay,
        )
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.model_optimizer, T_max=config.outer_steps, eta_min=0
        )

        # Tracking
        self.best_checkpoint = None
        self.early_stop_counter = 0

        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def _to_device(self, batch):
        return {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

    def _get_task_batch(self):
        try:
            batch = next(self.task_iter)
        except StopIteration:
            self.task_iter = iter(self.task_loader)
            batch = next(self.task_iter)
        return self._to_device(batch)

    def _get_probe_batch(self):
        try:
            batch = next(self.probe_iter)
        except StopIteration:
            self.probe_iter = iter(self.probe_loader)
            batch = next(self.probe_iter)
        return self._to_device(batch)

    def _get_anchor_batch(self):
        try:
            batch = next(self.anchor_iter)
        except StopIteration:
            self.anchor_iter = iter(self.anchor_loader)
            batch = next(self.anchor_iter)
        return self._to_device(batch)

    def _compute_lambda(self, step: int) -> float:
        return self.config.lambda_max * min(1.0, step / self.config.lambda_warmup_steps)

    def _reinit_probe(self):
        self.probe = LinearProbe(self.config.hidden_dim).to(self.device)
        self.probe_optimizer = torch.optim.SGD(
            self.probe.parameters(),
            lr=self.config.inner_lr,
            weight_decay=self.config.inner_weight_decay,
        )

    def _inner_loop(self):
        """Train the linear probe on detached activations."""
        self.model.eval()
        self.probe.train()

        batch = self._get_probe_batch()
        with torch.no_grad():
            acts = extract_activations(
                self.model,
                batch["input_ids"],
                batch["attention_mask"],
                layers=[self.config.target_layer],
            )
        h = acts[self.config.target_layer].to(self.device)
        labels = batch["labels"].to(self.device)

        probe_loss = train_linear_probe_inner(
            self.probe,
            self.probe_optimizer,
            h,
            labels,
            steps=self.config.inner_steps,
            batch_size=min(self.config.probe_batch_size, h.size(0)),
        )
        return probe_loss

    def _outer_step(self, step: int) -> dict[str, float]:
        """One outer-loop model update."""
        self.model.train()
        lam = self._compute_lambda(step)

        # --- Task loss: next-token prediction ---
        task_batch = self._get_task_batch()
        outputs = self.model(
            input_ids=task_batch["input_ids"],
            attention_mask=task_batch["attention_mask"],
            labels=task_batch["input_ids"],
        )
        l_task = outputs.loss

        # --- Anchor loss: KL divergence against frozen reference ---
        anchor_batch = self._get_anchor_batch()
        with torch.no_grad():
            ref_logits = self.ref_model(
                input_ids=anchor_batch["input_ids"],
                attention_mask=anchor_batch["attention_mask"],
            ).logits
        model_logits = self.model(
            input_ids=anchor_batch["input_ids"],
            attention_mask=anchor_batch["attention_mask"],
        ).logits

        # KL(ref || model) over vocabulary at each token position
        ref_probs = F.softmax(ref_logits, dim=-1)
        model_log_probs = F.log_softmax(model_logits, dim=-1)
        l_anchor = F.kl_div(model_log_probs, ref_probs, reduction="batchmean")

        # --- Probe adversarial loss ---
        # Activations MUST stay in the computation graph (no detach)
        probe_batch = self._get_probe_batch()
        model_outputs = self.model(
            input_ids=probe_batch["input_ids"],
            attention_mask=probe_batch["attention_mask"],
            output_hidden_states=True,
        )
        # +1 because hidden_states[0] is embedding output
        h = model_outputs.hidden_states[self.config.target_layer + 1]

        # Last-token pooling
        seq_lengths = probe_batch["attention_mask"].sum(dim=1) - 1
        h_pooled = h[torch.arange(h.size(0), device=self.device), seq_lengths]

        # Freeze probe weights for outer step
        self.probe.eval()
        for p in self.probe.parameters():
            p.requires_grad_(False)

        probe_logits = self.probe(h_pooled)
        labels = probe_batch["labels"].float().to(self.device)
        l_probe = F.binary_cross_entropy_with_logits(probe_logits, labels)

        # Re-enable probe gradients for next inner loop
        for p in self.probe.parameters():
            p.requires_grad_(True)

        # --- Combined loss ---
        loss = l_task + self.config.alpha * l_anchor - lam * l_probe

        self.model_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.model_optimizer.step()
        self.lr_scheduler.step()

        return {
            "loss": loss.item(),
            "l_task": l_task.item(),
            "l_anchor": l_anchor.item(),
            "l_probe": l_probe.item(),
            "lambda": lam,
            "lr": self.lr_scheduler.get_last_lr()[0],
        }

    def _sft_step(self):
        """Pure supervised fine-tuning step (no adversarial term) for stability."""
        self.model.train()
        batch = self._get_task_batch()
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["input_ids"],
        )
        self.model_optimizer.zero_grad()
        outputs.loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.model_optimizer.step()

    @torch.no_grad()
    def _eval_probe(self) -> dict[str, float]:
        """Train a fresh sklearn probe on held-out eval data and return AUC."""
        if self.eval_loader is None or self.eval_labels is None:
            return {}

        self.model.eval()
        acts = extract_all_activations(
            self.model, self.eval_loader,
            layers=[self.config.target_layer], device=self.device,
        )
        X = acts[self.config.target_layer].numpy()
        y = self.eval_labels

        # Split eval data 50/50 for probe train/test
        n = len(y)
        mid = n // 2
        probe = SklearnProbe()
        probe.fit(X[:mid], y[:mid])
        metrics = probe.evaluate(X[mid:], y[mid:])

        logger.info(
            f"  [eval] fresh probe AUC={metrics['auc']:.4f}  "
            f"acc={metrics['accuracy']:.4f}"
        )
        return {f"eval_{k}": v for k, v in metrics.items()}

    @torch.no_grad()
    def _log_generations(self, step: int) -> list[dict[str, str]]:
        """Generate completions from fixed prompts and log them."""
        if self.tokenizer is None:
            return []

        self.model.eval()
        generations = []

        for prompt in self.config.gen_prompts:
            input_ids = self.tokenizer(
                prompt, return_tensors="pt"
            ).input_ids.to(self.device)

            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.config.gen_max_new_tokens,
                do_sample=False,  # greedy for reproducibility
                pad_token_id=self.tokenizer.pad_token_id,
            )
            completion = self.tokenizer.decode(
                output_ids[0][input_ids.shape[1]:], skip_special_tokens=True
            )
            generations.append({"prompt": prompt, "completion": completion})

        # Log to console
        logger.info(f"  [gen] step {step} generations:")
        for g in generations:
            logger.info(f"    \"{g['prompt']}\" → \"{g['completion'][:80]}\"")

        return generations

    def _save_checkpoint(self, step: int, metrics: dict):
        path = Path(self.config.checkpoint_dir) / f"step_{step}"
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        torch.save(
            {"step": step, "metrics": metrics},
            path / "trainer_state.pt",
        )
        logger.info(f"Checkpoint saved at step {step}")

    def train(self):
        """Run the full adversarial training loop."""
        try:
            import wandb
            if self.config.use_wandb:
                wandb.init(project="activation-steganography", config=vars(self.config))
        except ImportError:
            logger.warning("wandb not installed, skipping logging")
            self.config.use_wandb = False

        last_good_state = None
        metrics_history = []
        eval_history = []
        gen_history = []

        # Log step-0 baselines before any training
        eval_metrics = self._eval_probe()
        generations = self._log_generations(0)
        eval_history.append({"step": 0, **eval_metrics})
        gen_history.append({"step": 0, "generations": generations})

        for step in tqdm(range(self.config.outer_steps), desc="Adversarial training"):
            # --- Inner loop ---
            probe_loss = self._inner_loop()

            # --- Outer step ---
            metrics = self._outer_step(step)
            metrics["probe_train_loss"] = probe_loss

            # --- NaN handling ---
            if any(torch.isnan(torch.tensor(v)) for v in metrics.values()):
                logger.warning(f"NaN detected at step {step}, rolling back")
                if last_good_state is not None:
                    self.model.load_state_dict(last_good_state)
                    self.config.lambda_max *= 0.5
                    logger.info(f"Halved lambda_max to {self.config.lambda_max}")
                self._reinit_probe()
                continue

            last_good_state = copy.deepcopy(self.model.state_dict())
            metrics_history.append(metrics)

            # --- SFT interleaving ---
            if step > 0 and step % self.config.sft_every == 0:
                self._sft_step()

            # --- Probe reinit ---
            if step > 0 and step % self.config.probe_reinit_every == 0:
                self._reinit_probe()
                logger.info(f"Probe reinitialized at step {step}")

            # --- Periodic eval: fresh probe + generation ---
            if step > 0 and step % self.config.eval_every == 0:
                eval_metrics = self._eval_probe()
                generations = self._log_generations(step)
                eval_history.append({"step": step, **eval_metrics})
                gen_history.append({"step": step, "generations": generations})
                metrics.update(eval_metrics)

            # --- Checkpoint ---
            if step > 0 and step % self.config.checkpoint_every == 0:
                self._save_checkpoint(step, metrics)

            if self.config.use_wandb:
                import wandb
                wandb.log(metrics, step=step)

            # --- Early stopping ---
            if len(metrics_history) >= self.config.early_stop_window:
                recent = metrics_history[-self.config.early_stop_window:]
                avg_probe_loss = sum(m["l_probe"] for m in recent) / len(recent)
                # High probe loss = probe is failing = we're succeeding
                if avg_probe_loss > 0.65:  # roughly corresponds to AUC < 0.55
                    logger.info(f"Early stopping at step {step}: probe loss consistently high")
                    break

        # Save final checkpoint
        self._save_checkpoint(step, metrics)

        # Final eval
        eval_metrics = self._eval_probe()
        generations = self._log_generations(step)
        eval_history.append({"step": step, "final": True, **eval_metrics})
        gen_history.append({"step": step, "final": True, "generations": generations})

        logger.info("Training complete")
        return {
            "metrics": metrics_history,
            "eval": eval_history,
            "generations": gen_history,
        }
