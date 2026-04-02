"""Evaluation suite for concept-hiding experiments.

Runs fresh linear probes (single-layer, per-layer sweep, all-layer concat),
perplexity measurement, and concept recovery attack.
"""

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.activation_extractor import extract_all_activations
from src.probes.probe import SklearnProbe

logger = logging.getLogger(__name__)


def evaluate_probes(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    target_layer: int,
    num_layers: int = 24,
    device: str = "cuda",
) -> dict[str, dict[str, float]]:
    """Train fresh linear probes and evaluate.

    Returns results for:
    - single_layer: probe at target layer L
    - per_layer_best: best probe across all layers
    - all_layer_concat: probe on concatenation of all layers
    """
    results = {}

    # --- Single-layer probe at target layer L ---
    logger.info(f"Evaluating single-layer probe at layer {target_layer}")
    train_acts = extract_all_activations(model, train_loader, layers=[target_layer], device=device)
    test_acts = extract_all_activations(model, test_loader, layers=[target_layer], device=device)

    X_train = train_acts[target_layer].numpy()
    X_test = test_acts[target_layer].numpy()

    probe = SklearnProbe()
    probe.fit(X_train, train_labels)
    results["single_layer"] = probe.evaluate(X_test, test_labels)
    results["single_layer"]["layer"] = target_layer
    logger.info(f"Single-layer (L={target_layer}): {results['single_layer']}")

    # --- Per-layer sweep ---
    logger.info("Running per-layer probe sweep")
    all_layers = list(range(num_layers))
    train_acts_all = extract_all_activations(model, train_loader, layers=all_layers, device=device)
    test_acts_all = extract_all_activations(model, test_loader, layers=all_layers, device=device)

    best_auc = 0.0
    best_layer = 0
    per_layer_results = {}

    for layer in all_layers:
        X_tr = train_acts_all[layer].numpy()
        X_te = test_acts_all[layer].numpy()

        probe = SklearnProbe()
        probe.fit(X_tr, train_labels)
        layer_metrics = probe.evaluate(X_te, test_labels)
        per_layer_results[layer] = layer_metrics

        if layer_metrics["auc"] > best_auc:
            best_auc = layer_metrics["auc"]
            best_layer = layer

    results["per_layer"] = per_layer_results
    results["per_layer_best"] = {**per_layer_results[best_layer], "layer": best_layer}
    logger.info(f"Per-layer best (L={best_layer}): AUC={best_auc:.4f}")

    # --- All-layer concat probe ---
    logger.info("Evaluating all-layer concat probe")
    X_train_concat = np.concatenate(
        [train_acts_all[l].numpy() for l in all_layers], axis=1
    )
    X_test_concat = np.concatenate(
        [test_acts_all[l].numpy() for l in all_layers], axis=1
    )

    probe = SklearnProbe()
    probe.fit(X_train_concat, train_labels)
    results["all_layer_concat"] = probe.evaluate(X_test_concat, test_labels)
    logger.info(f"All-layer concat: {results['all_layer_concat']}")

    return results


@torch.no_grad()
def evaluate_perplexity(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
) -> float:
    """Compute perplexity on a dataset."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )

        # Count only non-padding tokens
        num_tokens = attention_mask.sum().item()
        total_loss += outputs.loss.item() * num_tokens
        total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    return float(np.exp(avg_loss))


def concept_recovery_attack(
    model: nn.Module,
    probe_train_loader: DataLoader,
    probe_test_loader: DataLoader,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    target_layer: int,
    recovery_steps: int = 100,
    recovery_lr: float = 2e-5,
    eval_every: int = 10,
    device: str = "cuda",
) -> list[dict[str, float]]:
    """Concept recovery attack: fine-tune the defended model on sentiment data
    and track how quickly probe AUC recovers.

    Returns list of {step, auc} dicts.
    """
    import copy

    # Work on a copy so we don't modify the defended model
    attack_model = copy.deepcopy(model).to(device)
    attack_model.train()

    optimizer = torch.optim.AdamW(attack_model.parameters(), lr=recovery_lr)
    train_iter = iter(probe_train_loader)

    recovery_curve = []

    for step in range(recovery_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(probe_train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = attack_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )

        optimizer.zero_grad()
        outputs.loss.backward()
        optimizer.step()

        if step % eval_every == 0:
            attack_model.eval()
            train_acts = extract_all_activations(
                attack_model, probe_train_loader, layers=[target_layer], device=device
            )
            test_acts = extract_all_activations(
                attack_model, probe_test_loader, layers=[target_layer], device=device
            )

            probe = SklearnProbe()
            probe.fit(train_acts[target_layer].numpy(), train_labels)
            metrics = probe.evaluate(test_acts[target_layer].numpy(), test_labels)

            recovery_curve.append({"step": step, **metrics})
            logger.info(f"Recovery step {step}: AUC={metrics['auc']:.4f}")
            attack_model.train()

    return recovery_curve
