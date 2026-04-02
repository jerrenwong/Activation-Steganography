"""End-to-end overnight run: Baselines → Train → Eval.

Usage:
    python -m scripts.run_all [--config configs/default.yaml]

Writes all results to results/ directory. Designed to run unattended overnight.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import yaml
from torch.utils.data import DataLoader

from src.data.dataset_builder import get_tokenizer, load_imdb, load_openwebtext, load_sst2
from src.evaluation.eval_suite import concept_recovery_attack
from src.models.activation_extractor import extract_all_activations, load_model
from src.probes.probe import SklearnProbe
from src.training.adversarial_trainer import AdversarialTrainer, TrainConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("results/run_all.log"),
    ],
)
logger = logging.getLogger(__name__)


def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved {path}")


def probe_sweep(model, sst2, config):
    """Per-layer linear probe sweep. Returns target layer and per-layer AUCs."""
    logger.info("=" * 60)
    logger.info("BASELINES: per-layer probe sweep")
    logger.info("=" * 60)

    num_layers = config["model"]["num_layers"]
    all_layers = list(range(num_layers))

    train_loader = DataLoader(sst2["train"], batch_size=64, shuffle=False)
    val_loader = DataLoader(sst2["val"], batch_size=64, shuffle=False)
    test_loader = DataLoader(sst2["test"], batch_size=64, shuffle=False)

    train_labels = np.array([sst2["train"][i]["labels"] for i in range(len(sst2["train"]))])
    val_labels = np.array([sst2["val"][i]["labels"] for i in range(len(sst2["val"]))])
    test_labels = np.array([sst2["test"][i]["labels"] for i in range(len(sst2["test"]))])

    logger.info(f"Extracting activations at {num_layers} layers")
    train_acts = extract_all_activations(model, train_loader, layers=all_layers, device="cuda")
    val_acts = extract_all_activations(model, val_loader, layers=all_layers, device="cuda")
    test_acts = extract_all_activations(model, test_loader, layers=all_layers, device="cuda")

    per_layer = {}
    best_val_auc = 0.0
    best_layer = 0

    for layer in all_layers:
        probe = SklearnProbe(
            C=config["probe_sweep"]["probe_C"],
            max_iter=config["probe_sweep"]["probe_max_iter"],
        )
        probe.fit(train_acts[layer].numpy(), train_labels)
        val_metrics = probe.evaluate(val_acts[layer].numpy(), val_labels)
        test_metrics = probe.evaluate(test_acts[layer].numpy(), test_labels)
        per_layer[layer] = {"val": val_metrics, "test": test_metrics}
        logger.info(
            f"  Layer {layer:2d}: val AUC={val_metrics['auc']:.4f}  "
            f"test AUC={test_metrics['auc']:.4f}"
        )
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_layer = layer

    target_layer = best_layer
    logger.info(f"Target layer: {target_layer} (val AUC={best_val_auc:.4f})")

    result = {
        "target_layer": target_layer,
        "single_layer_auc": per_layer[target_layer]["test"]["auc"],
        "per_layer": {str(k): v for k, v in per_layer.items()},
    }
    save_json(result, "results/baselines.json")

    return target_layer, result, train_labels, test_labels


def train(model, tokenizer, config, target_layer):
    """Adversarial training."""
    logger.info("=" * 60)
    logger.info("TRAINING: adversarial concept hiding")
    logger.info("=" * 60)

    tc = config["training"]

    sst2 = load_sst2(tokenizer, max_length=config["data"]["sst2_max_length"])

    logger.info("Loading OpenWebText (task + anchor)")
    owt = load_openwebtext(
        tokenizer,
        max_length=config["data"]["owt_max_length"],
        num_task=config["data"]["owt_num_task"],
        num_anchor=config["data"]["owt_num_anchor"],
    )

    task_loader = DataLoader(owt["task"], batch_size=tc["task_batch_size"], shuffle=True)
    probe_loader = DataLoader(sst2["train"], batch_size=tc["probe_batch_size"], shuffle=True)
    anchor_loader = DataLoader(owt["anchor"], batch_size=tc["task_batch_size"], shuffle=True)

    # Held-out eval data for periodic probe evaluation
    eval_loader = DataLoader(sst2["val"], batch_size=64, shuffle=False)
    eval_labels = np.array([sst2["val"][i]["labels"] for i in range(len(sst2["val"]))])

    train_config = TrainConfig(
        outer_lr=tc["outer_lr"],
        outer_weight_decay=tc["outer_weight_decay"],
        outer_steps=tc["outer_steps"],
        task_batch_size=tc["task_batch_size"],
        max_grad_norm=tc["max_grad_norm"],
        alpha=tc["alpha"],
        lambda_max=tc["lambda_max"],
        lambda_warmup_steps=tc["lambda_warmup_steps"],
        inner_lr=tc["inner_lr"],
        inner_weight_decay=tc["inner_weight_decay"],
        inner_steps=tc["inner_steps"],
        probe_batch_size=tc["probe_batch_size"],
        probe_reinit_every=tc["probe_reinit_every"],
        sft_every=tc["sft_every"],
        checkpoint_every=tc["checkpoint_every"],
        target_layer=target_layer,
        hidden_dim=config["model"]["hidden_dim"],
        device=tc["device"],
        dtype=tc["dtype"],
        checkpoint_dir=tc["checkpoint_dir"],
        use_wandb=tc["use_wandb"],
    )

    trainer = AdversarialTrainer(
        model=model,
        task_loader=task_loader,
        probe_loader=probe_loader,
        anchor_loader=anchor_loader,
        config=train_config,
        tokenizer=tokenizer,
        eval_loader=eval_loader,
        eval_labels=eval_labels,
    )

    results = trainer.train()
    save_json(results["metrics"], "results/training_metrics.json")
    save_json(results["eval"], "results/training_eval.json")
    save_json(results["generations"], "results/training_generations.json")

    return trainer.model


def evaluate(model, tokenizer, config, target_layer, baseline_results):
    """Post-training: fresh per-layer probes, IMDB generalization, recovery attack."""
    logger.info("=" * 60)
    logger.info("EVALUATION: defended model")
    logger.info("=" * 60)

    ec = config["evaluation"]
    num_layers = config["model"]["num_layers"]
    all_layers = list(range(num_layers))

    sst2 = load_sst2(tokenizer, max_length=config["data"]["sst2_max_length"])
    train_loader = DataLoader(sst2["train"], batch_size=64, shuffle=False)
    test_loader = DataLoader(sst2["test"], batch_size=64, shuffle=False)
    train_labels = np.array([sst2["train"][i]["labels"] for i in range(len(sst2["train"]))])
    test_labels = np.array([sst2["test"][i]["labels"] for i in range(len(sst2["test"]))])

    # --- Fresh per-layer probe sweep on defended model ---
    logger.info("Fresh per-layer probe sweep on defended model")
    train_acts = extract_all_activations(model, train_loader, layers=all_layers, device="cuda")
    test_acts = extract_all_activations(model, test_loader, layers=all_layers, device="cuda")

    per_layer = {}
    best_auc = 0.0
    best_layer = 0
    for layer in all_layers:
        probe = SklearnProbe()
        probe.fit(train_acts[layer].numpy(), train_labels)
        metrics = probe.evaluate(test_acts[layer].numpy(), test_labels)
        per_layer[layer] = metrics
        logger.info(f"  Layer {layer:2d}: AUC={metrics['auc']:.4f}")
        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            best_layer = layer

    # --- IMDB cross-dataset generalization ---
    logger.info("IMDB cross-dataset generalization")
    imdb = load_imdb(tokenizer, max_length=config["data"]["imdb_max_length"])
    imdb_loader = DataLoader(imdb["test"], batch_size=64, shuffle=False)
    imdb_labels = np.array([imdb["test"][i]["labels"] for i in range(len(imdb["test"]))])

    imdb_acts = extract_all_activations(model, imdb_loader, layers=[target_layer], device="cuda")
    imdb_probe = SklearnProbe()
    imdb_probe.fit(train_acts[target_layer].numpy(), train_labels)
    imdb_metrics = imdb_probe.evaluate(imdb_acts[target_layer].numpy(), imdb_labels)
    logger.info(f"  IMDB AUC (probe trained on SST-2): {imdb_metrics['auc']:.4f}")

    # --- Concept recovery attack ---
    logger.info("Concept recovery attack")
    recovery_curve = concept_recovery_attack(
        model=model,
        probe_train_loader=train_loader,
        probe_test_loader=test_loader,
        train_labels=train_labels,
        test_labels=test_labels,
        target_layer=target_layer,
        recovery_steps=ec["recovery_steps"],
        recovery_lr=ec["recovery_lr"],
        eval_every=ec["recovery_eval_every"],
        device="cuda",
    )

    # --- Save and summarize ---
    bl = baseline_results
    results = {
        "target_layer_auc": {
            "baseline": bl["single_layer_auc"],
            "defended": per_layer[target_layer]["auc"],
        },
        "best_layer_post_defense": {"layer": best_layer, "auc": best_auc},
        "per_layer": {str(k): v for k, v in per_layer.items()},
        "imdb_cross_dataset": imdb_metrics,
        "recovery_curve": recovery_curve,
    }
    save_json(results, "results/eval.json")

    logger.info("")
    logger.info("=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Target layer (L={target_layer}) AUC:  {bl['single_layer_auc']:.4f} → {per_layer[target_layer]['auc']:.4f}")
    logger.info(f"  Best layer post-defense:     L={best_layer}, AUC={best_auc:.4f}")
    logger.info(f"  IMDB cross-dataset AUC:      {imdb_metrics['auc']:.4f}")
    logger.info(f"  Recovery (step 0):           AUC={recovery_curve[0]['auc']:.4f}")
    logger.info(f"  Recovery (step {recovery_curve[-1]['step']}):         AUC={recovery_curve[-1]['auc']:.4f}")
    logger.info("=" * 60)

    return results


def main(config_path: str = "configs/default.yaml"):
    Path("results").mkdir(exist_ok=True)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    start_time = time.time()

    logger.info(f"Loading model: {config['model']['name']}")
    tokenizer = get_tokenizer(config["model"]["name"])
    model = load_model(config["model"]["name"], device="cuda")
    model.eval()

    logger.info("Loading SST-2")
    sst2 = load_sst2(tokenizer, max_length=config["data"]["sst2_max_length"])

    # Baselines: per-layer probe sweep
    target_layer, baseline_results, _, _ = probe_sweep(model, sst2, config)

    # Training: adversarial concept hiding
    model = train(model, tokenizer, config, target_layer)

    # Evaluation: fresh probes, IMDB, recovery attack
    evaluate(model, tokenizer, config, target_layer, baseline_results)

    elapsed = time.time() - start_time
    logger.info(f"Total runtime: {elapsed / 3600:.1f} hours")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
