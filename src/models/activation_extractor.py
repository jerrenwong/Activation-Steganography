"""Activation extraction from Pythia-410M.

Provides forward pass through the model and extraction of residual stream
activations at specified layers, with last-token pooling.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


def load_model(
    model_name: str = "EleutherAI/pythia-410m",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
    )
    return model


def extract_activations(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layers: list[int] | None = None,
    pool: str = "last",
) -> dict[int, torch.Tensor]:
    """Extract residual stream activations at specified layers.

    Args:
        model: Pythia model (or any HF causal LM).
        input_ids: (batch, seq_len) token IDs.
        attention_mask: (batch, seq_len) attention mask.
        layers: Which layers to extract. None = all layers.
        pool: 'last' for last-token pooling, 'mean' for mean pooling.

    Returns:
        Dict mapping layer index to tensor of shape (batch, hidden_dim).
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )

    # hidden_states is a tuple of (num_layers + 1) tensors (includes embedding layer)
    # hidden_states[0] = embedding output, hidden_states[i] = layer i output
    hidden_states = outputs.hidden_states

    if layers is None:
        layers = list(range(len(hidden_states) - 1))  # exclude embedding layer 0

    result = {}
    for layer_idx in layers:
        # +1 because hidden_states[0] is the embedding layer
        h = hidden_states[layer_idx + 1]  # (batch, seq_len, hidden_dim)

        if pool == "last":
            # Find the index of the last real token (before padding)
            seq_lengths = attention_mask.sum(dim=1) - 1  # (batch,)
            pooled = h[torch.arange(h.size(0), device=h.device), seq_lengths]
        elif pool == "mean":
            mask_expanded = attention_mask.unsqueeze(-1).float()  # (batch, seq, 1)
            pooled = (h * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            raise ValueError(f"Unknown pool method: {pool}")

        result[layer_idx] = pooled  # (batch, hidden_dim)

    return result


@torch.no_grad()
def extract_all_activations(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    layers: list[int] | None = None,
    device: str = "cuda",
) -> dict[int, torch.Tensor]:
    """Extract activations for an entire dataset.

    Returns dict mapping layer index to tensor of shape (N, hidden_dim)
    where N is the total number of examples.
    """
    model.eval()
    collected: dict[int, list[torch.Tensor]] = {}

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        acts = extract_activations(model, input_ids, attention_mask, layers=layers)

        for layer_idx, tensor in acts.items():
            if layer_idx not in collected:
                collected[layer_idx] = []
            collected[layer_idx].append(tensor.cpu().float())

    return {k: torch.cat(v, dim=0) for k, v in collected.items()}
