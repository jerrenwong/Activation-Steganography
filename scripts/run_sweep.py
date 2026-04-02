"""Phase 0: Probe sweep across all layers to find the target layer.

Extracts activations at every layer of Pythia-410M on SST-2, trains a linear
probe at each layer, and reports AUC. The layer with the highest AUC becomes
the target layer L for adversarial training.

Usage:
    python -m scripts.run_sweep [--config configs/default.yaml]
"""

import argparse
import json
import logging

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset_builder import get_tokenizer, load_sst2
from src.models.activation_extractor import extract_all_activations, load_model
from src.probes.probe import SklearnProbe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main(config_path: str = "configs/default.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["name"]
    num_layers = config["model"]["num_layers"]

    # Load model and tokenizer
    logger.info(f"Loading model: {model_name}")
    tokenizer = get_tokenizer(model_name)
    model = load_model(model_name, device="cuda")
    model.eval()

    # Load SST-2
    logger.info("Loading SST-2 dataset")
    sst2 = load_sst2(tokenizer, max_length=config["data"]["sst2_max_length"])

    train_loader = DataLoader(sst2["train"], batch_size=64, shuffle=False)
    val_loader = DataLoader(sst2["val"], batch_size=64, shuffle=False)
    test_loader = DataLoader(sst2["test"], batch_size=64, shuffle=False)

    train_labels = [sst2["train"][i]["labels"] for i in range(len(sst2["train"]))]
    val_labels = [sst2["val"][i]["labels"] for i in range(len(sst2["val"]))]
    test_labels = [sst2["test"][i]["labels"] for i in range(len(sst2["test"]))]

    import numpy as np
    train_labels = np.array(train_labels)
    val_labels = np.array(val_labels)
    test_labels = np.array(test_labels)

    # Extract activations at all layers
    logger.info(f"Extracting activations at {num_layers} layers")
    all_layers = list(range(num_layers))

    train_acts = extract_all_activations(model, train_loader, layers=all_layers, device="cuda")
    val_acts = extract_all_activations(model, val_loader, layers=all_layers, device="cuda")
    test_acts = extract_all_activations(model, test_loader, layers=all_layers, device="cuda")

    # Train probes at each layer
    results = {}
    best_val_auc = 0.0
    best_layer = 0

    for layer in all_layers:
        X_train = train_acts[layer].numpy()
        X_val = val_acts[layer].numpy()
        X_test = test_acts[layer].numpy()

        probe = SklearnProbe(
            C=config["probe_sweep"]["probe_C"],
            max_iter=config["probe_sweep"]["probe_max_iter"],
        )
        probe.fit(X_train, train_labels)

        val_metrics = probe.evaluate(X_val, val_labels)
        test_metrics = probe.evaluate(X_test, test_labels)

        results[layer] = {
            "val": val_metrics,
            "test": test_metrics,
        }

        logger.info(
            f"Layer {layer:2d}: val AUC={val_metrics['auc']:.4f}  "
            f"test AUC={test_metrics['auc']:.4f}  "
            f"test acc={test_metrics['accuracy']:.4f}"
        )

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_layer = layer

    logger.info(f"\nBest layer: {best_layer} (val AUC={best_val_auc:.4f})")
    logger.info(f"Test AUC at best layer: {results[best_layer]['test']['auc']:.4f}")

    # Save results
    output = {
        "best_layer": best_layer,
        "best_val_auc": best_val_auc,
        "per_layer": {str(k): v for k, v in results.items()},
    }
    with open("probe_sweep_results.json", "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to probe_sweep_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
