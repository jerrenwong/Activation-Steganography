"""Dataset builder for concept-hiding experiments.

Loads SST-2 (sentiment labels), IMDB (generalization eval), and OpenWebText
(task/anchor data). Uses hash-based splitting for strict disjointness across
experiment phases.
"""

import hashlib
from dataclasses import dataclass

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def _hash_split(text: str, salt: str, boundaries: tuple[float, ...] = (0.7, 0.85)) -> str:
    """Deterministic hash-based split assignment.

    Returns 'train', 'val', or 'test' based on hash of (text + salt).
    boundaries=(0.7, 0.85) gives 70/15/15 split.
    """
    h = int(hashlib.md5((text + salt).encode()).hexdigest(), 16) % 10000
    frac = h / 10000
    if frac < boundaries[0]:
        return "train"
    elif len(boundaries) > 1 and frac < boundaries[1]:
        return "val"
    return "test"


@dataclass
class SentimentExample:
    text: str
    label: int  # 0=negative, 1=positive
    split: str  # train/val/test


class TokenizedDataset(Dataset):
    """Wraps tokenized examples for DataLoader use."""

    def __init__(self, encodings: dict, labels: list[int] | None = None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


def load_sst2(
    tokenizer: AutoTokenizer,
    max_length: int = 128,
    split_salt: str = "sst2_v1",
) -> dict[str, TokenizedDataset]:
    """Load SST-2 with hash-based train/val/test split.

    Returns dict with keys 'train', 'val', 'test', each a TokenizedDataset.
    """
    ds = load_dataset("glue", "sst2", split="train")

    splits: dict[str, list] = {"train": [], "val": [], "test": []}
    labels_by_split: dict[str, list] = {"train": [], "val": [], "test": []}

    for example in ds:
        text = example["sentence"]
        label = example["label"]
        split = _hash_split(text, split_salt)
        splits[split].append(text)
        labels_by_split[split].append(label)

    result = {}
    for split_name in ("train", "val", "test"):
        texts = splits[split_name]
        labels = labels_by_split[split_name]
        encodings = tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        result[split_name] = TokenizedDataset(encodings, labels)

    return result


def load_imdb(
    tokenizer: AutoTokenizer,
    max_length: int = 256,
) -> dict[str, TokenizedDataset]:
    """Load IMDB test split for generalization evaluation."""
    ds = load_dataset("imdb", split="test")

    texts = [ex["text"] for ex in ds]
    labels = [ex["label"] for ex in ds]

    encodings = tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return {"test": TokenizedDataset(encodings, labels)}


def load_openwebtext(
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    num_task: int = 40_000,
    num_anchor: int = 2_000,
    split_salt: str = "owt_v1",
) -> dict[str, TokenizedDataset]:
    """Load OpenWebText subset for task loss and anchor loss.

    Returns dict with 'task' and 'anchor' splits.
    """
    total_needed = num_task + num_anchor
    # Load a streaming subset — OpenWebText is large
    ds = load_dataset("openwebtext", split="train", streaming=True)

    task_texts = []
    anchor_texts = []

    for example in ds:
        text = example["text"]
        if len(text) < 100:
            continue
        split = _hash_split(text, split_salt, boundaries=(num_task / total_needed,))
        if split == "train" and len(task_texts) < num_task:
            task_texts.append(text)
        elif split != "train" and len(anchor_texts) < num_anchor:
            anchor_texts.append(text)

        if len(task_texts) >= num_task and len(anchor_texts) >= num_anchor:
            break

    result = {}
    for name, texts in [("task", task_texts), ("anchor", anchor_texts)]:
        encodings = tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        result[name] = TokenizedDataset(encodings)

    return result


def get_tokenizer(model_name: str = "EleutherAI/pythia-410m") -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
