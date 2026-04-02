"""Phase 2: Adversarial training.

Runs the minimax training loop to make sentiment undetectable by linear probes
at the target layer while preserving model capabilities.

Usage:
    python -m scripts.run_train --target-layer 15 [--config configs/default.yaml]
"""

import argparse
import logging

import yaml
from torch.utils.data import DataLoader

from src.data.dataset_builder import get_tokenizer, load_openwebtext, load_sst2
from src.models.activation_extractor import load_model
from src.training.adversarial_trainer import AdversarialTrainer, TrainConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main(config_path: str = "configs/default.yaml", target_layer: int = 15):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["name"]
    tc = config["training"]

    # Load model
    logger.info(f"Loading model: {model_name}")
    tokenizer = get_tokenizer(model_name)
    model = load_model(model_name, device=tc["device"])

    # Load datasets
    logger.info("Loading SST-2 (probe labels)")
    sst2 = load_sst2(tokenizer, max_length=config["data"]["sst2_max_length"])

    logger.info("Loading OpenWebText (task + anchor)")
    owt = load_openwebtext(
        tokenizer,
        max_length=config["data"]["owt_max_length"],
        num_task=config["data"]["owt_num_task"],
        num_anchor=config["data"]["owt_num_anchor"],
    )

    # Data loaders
    task_loader = DataLoader(owt["task"], batch_size=tc["task_batch_size"], shuffle=True)
    probe_loader = DataLoader(sst2["train"], batch_size=tc["probe_batch_size"], shuffle=True)
    anchor_loader = DataLoader(owt["anchor"], batch_size=tc["task_batch_size"], shuffle=True)

    # Training config
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

    # Train
    trainer = AdversarialTrainer(
        model=model,
        task_loader=task_loader,
        probe_loader=probe_loader,
        anchor_loader=anchor_loader,
        config=train_config,
    )

    logger.info(f"Starting adversarial training (target layer={target_layer})")
    trainer.train()
    logger.info("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--target-layer", type=int, required=True, help="Target layer from Phase 0 sweep")
    args = parser.parse_args()
    main(args.config, args.target_layer)
