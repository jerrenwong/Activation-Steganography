"""Smoke test: skip Phase 0 (use saved baselines), use synthetic OWT, run tiny training + eval.

Verifies the full pipeline won't crash before committing to an overnight run.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

from scripts.run_all import train, evaluate
from src.data.dataset_builder import TokenizedDataset, get_tokenizer
from src.models.activation_extractor import load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("results/smoke.log"),
    ],
)
logger = logging.getLogger(__name__)


def fake_load_openwebtext(tokenizer, max_length=512, num_task=200, num_anchor=100, split_salt="owt_v1"):
    """Generate synthetic OWT data to skip the massive download."""
    result = {}
    for name, n in [("task", num_task), ("anchor", num_anchor)]:
        # Random token IDs within vocab range
        input_ids = torch.randint(0, tokenizer.vocab_size, (n, max_length))
        attention_mask = torch.ones_like(input_ids)
        encodings = {"input_ids": input_ids, "attention_mask": attention_mask}
        result[name] = TokenizedDataset(encodings)
    return result


def main(config_path: str):
    Path("results").mkdir(exist_ok=True)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load saved baselines
    with open("results/baselines.json") as f:
        baseline_results = json.load(f)

    target_layer = baseline_results["target_layer"]
    logger.info(f"Using saved baselines: target_layer={target_layer}, AUC={baseline_results['single_layer_auc']:.4f}")

    start = time.time()

    logger.info(f"Loading model: {config['model']['name']}")
    tokenizer = get_tokenizer(config["model"]["name"])
    model = load_model(config["model"]["name"], device="cuda")

    # Phase 1: Training (with fake OWT to skip download)
    with patch("scripts.run_all.load_openwebtext", side_effect=fake_load_openwebtext):
        model = train(model, tokenizer, config, target_layer)

    # Phase 2: Evaluation
    evaluate(model, tokenizer, config, target_layer, baseline_results)

    logger.info(f"Smoke test completed in {time.time() - start:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke_test.yaml")
    args = parser.parse_args()
    main(args.config)
