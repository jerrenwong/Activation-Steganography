"""Phase 3: Evaluation of the defended model.

Trains fresh linear probes on the defended model's activations, measures
perplexity, and runs the concept recovery attack.

Usage:
    python -m scripts.run_eval --checkpoint checkpoints/step_5000 --target-layer 15 [--config configs/default.yaml]
"""

import argparse
import json
import logging

import numpy as np
import yaml
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from src.data.dataset_builder import get_tokenizer, load_openwebtext, load_sst2
from src.evaluation.eval_suite import (
    concept_recovery_attack,
    evaluate_perplexity,
    evaluate_probes,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main(
    config_path: str = "configs/default.yaml",
    checkpoint_path: str = "checkpoints/step_5000",
    target_layer: int = 15,
):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["name"]
    ec = config["evaluation"]

    # Load defended model from checkpoint
    logger.info(f"Loading defended model from {checkpoint_path}")
    import torch
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    tokenizer = get_tokenizer(model_name)

    # Load datasets
    logger.info("Loading SST-2")
    sst2 = load_sst2(tokenizer, max_length=config["data"]["sst2_max_length"])

    train_loader = DataLoader(sst2["train"], batch_size=64, shuffle=False)
    test_loader = DataLoader(sst2["test"], batch_size=64, shuffle=False)

    train_labels = np.array([sst2["train"][i]["labels"] for i in range(len(sst2["train"]))])
    test_labels = np.array([sst2["test"][i]["labels"] for i in range(len(sst2["test"]))])

    # --- Probe evaluation ---
    logger.info("Evaluating fresh linear probes")
    probe_results = evaluate_probes(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        train_labels=train_labels,
        test_labels=test_labels,
        target_layer=target_layer,
        num_layers=config["model"]["num_layers"],
        device="cuda",
    )

    # --- Perplexity ---
    logger.info("Evaluating perplexity")
    logger.info("Loading OpenWebText for perplexity eval")
    owt = load_openwebtext(
        tokenizer,
        max_length=config["data"]["owt_max_length"],
        num_task=1000,
        num_anchor=0,
        split_salt="owt_eval_v1",  # different salt for eval data
    )
    ppl_loader = DataLoader(owt["task"], batch_size=8, shuffle=False)
    ppl = evaluate_perplexity(model, ppl_loader, device="cuda")
    logger.info(f"Perplexity: {ppl:.2f}")

    # --- Concept recovery attack ---
    logger.info("Running concept recovery attack")
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

    # --- Save results ---
    results = {
        "checkpoint": checkpoint_path,
        "target_layer": target_layer,
        "probes": {
            "single_layer": probe_results["single_layer"],
            "per_layer_best": probe_results["per_layer_best"],
            "all_layer_concat": probe_results["all_layer_concat"],
        },
        "perplexity": ppl,
        "recovery_curve": recovery_curve,
    }

    output_path = f"eval_results_{checkpoint_path.replace('/', '_')}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Single-layer probe (L={target_layer}):  AUC={probe_results['single_layer']['auc']:.4f}")
    print(f"Best per-layer probe (L={probe_results['per_layer_best']['layer']}): AUC={probe_results['per_layer_best']['auc']:.4f}")
    print(f"All-layer concat probe:       AUC={probe_results['all_layer_concat']['auc']:.4f}")
    print(f"Perplexity:                    {ppl:.2f}")
    print(f"Recovery (step 0):             AUC={recovery_curve[0]['auc']:.4f}")
    print(f"Recovery (step {recovery_curve[-1]['step']}):           AUC={recovery_curve[-1]['auc']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to defended model checkpoint")
    parser.add_argument("--target-layer", type=int, required=True)
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.target_layer)
