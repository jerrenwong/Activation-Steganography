"""Linear probes for concept detection.

Two implementations:
1. LinearProbe (nn.Module): Differentiable, used in the adversarial training loop
   where gradients must flow through the probe back into the model.
2. SklearnProbe: Wrapper around LogisticRegression for Phase 0 sweep and
   evaluation (faster, closed-form solution, no hyperparameter tuning needed).
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class LinearProbe(nn.Module):
    """Differentiable linear probe for adversarial training.

    A single linear layer: h (hidden_dim,) -> logit (1,).
    Used in the inner/outer loop where gradients must flow through.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            h: (batch, hidden_dim) activations.

        Returns:
            (batch,) logits (pre-sigmoid).
        """
        return self.linear(h.to(self.linear.weight.dtype)).squeeze(-1)


class SklearnProbe:
    """Sklearn LogisticRegression wrapper for sweep and evaluation.

    Fits on numpy arrays of pre-extracted activations. No gradient flow.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        self.model = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        proba = self.predict_proba(X)[:, 1]
        preds = (proba > 0.5).astype(int)
        return {
            "auc": roc_auc_score(y, proba),
            "accuracy": accuracy_score(y, preds),
            "f1": f1_score(y, preds),
        }


def train_linear_probe_inner(
    probe: LinearProbe,
    optimizer: torch.optim.Optimizer,
    activations: torch.Tensor,
    labels: torch.Tensor,
    steps: int = 5,
    batch_size: int = 256,
) -> float:
    """Train the differentiable linear probe for `steps` inner-loop steps.

    Args:
        probe: LinearProbe module.
        optimizer: SGD optimizer for probe parameters.
        activations: (N, hidden_dim) detached activations from the model.
        labels: (N,) binary sentiment labels.
        steps: Number of inner-loop steps.
        batch_size: Batch size for inner-loop updates.

    Returns:
        Final probe loss (float).
    """
    probe.train()
    n = activations.size(0)
    loss_fn = nn.BCEWithLogitsLoss()
    final_loss = 0.0

    for _ in range(steps):
        idx = torch.randint(0, n, (batch_size,))
        h = activations[idx]
        y = labels[idx].float()

        logits = probe(h)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    return final_loss
