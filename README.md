# Activation Steganography: Hiding Sentiment from Neural Probes

Fine-tune Pythia-410M so that **linear probes** trained on its internal activations cannot detect sentiment (positive vs. negative), while the model retains its sentiment-processing capabilities. A proof-of-concept for activation-level steganography with implications for AI safety monitoring.

## Motivation

Activation-based monitors are a promising approach for detecting undesirable model behavior (deception, harmful intent) by training classifiers on internal representations. But if a model can learn to hide a concept from such probes while retaining capability, these monitors may provide false assurance. This project studies that threat by:

1. Finding the layer where sentiment is most linearly detectable in Pythia-410M
2. Training the model adversarially to make sentiment linearly undetectable at that layer
3. Evaluating whether fresh linear probes can still recover sentiment

## Formulation

A minimax game between the model and a probe family F:

```
min_θ [ L_task(θ) + α · L_anchor(θ, θ_ref) - λ · L_probe(φ*, h_l(θ)) ]

where φ* = argmin_φ L_probe(φ, h_l(θ))    (best probe from family F)
```

- **L_task**: next-token prediction loss on mixed text (preserves general capability)
- **L_anchor**: KL(M_θ ‖ M_ref) on held-out data (prevents catastrophic drift)
- **L_probe**: cross-entropy of the best probe on (activations, sentiment labels)
- The **negative sign** on L_probe means the model is rewarded when the probe fails

F is a pluggable probe family — the training loop is identical regardless of what F contains. For this experiment, **F = linear classifiers** (logistic regression). The inner loop trains the best linear probe on current activations. The outer loop updates model weights to simultaneously maintain task performance and maximize that probe's loss. Future work can extend F to MLPs, attention probes, or ensembles.

## Model

**Pythia-410M** (EleutherAI)
- 24 transformer layers, hidden dimension 1024
- Most-studied model family for probing and interpretability
- LEACE, INLP, and RLACE all tested on Pythia
- Full fine-tuning uses ~8–10GB VRAM (0.8GB params + 0.8GB ref + 3.3GB optimizer + 1.6GB gradients + ~3GB activations)

## Experiment Design

### Phase 0: Probe Sweep — Find the Target Layer

**Goal**: Identify which layer encodes sentiment most strongly.

1. **Dataset**: SST-2 training set (~67k labeled examples, binary positive/negative)
2. **Activation extraction**: Forward pass through Pythia-410M, extract residual stream at the last token position for all 24 layers. Store as float16 tensors of shape `(N, 1024)` per layer (~130MB per layer for training set)
3. **Probe**: `LogisticRegression(C=1.0, max_iter=1000)` via sklearn at each layer
4. **Metrics**: Validation AUC-ROC, accuracy, F1 (70/15/15 train/val/test split)
5. **Layer selection**: Target layer **L** = layer with highest linear probe validation AUC. Literature suggests layers 12–18 (50–75% depth)

**Expected outcome**: Linear probe AUC > 0.80 at peak layer.

**Compute**: ~2 hours on A100-80GB.

### Phase 1: Baselines

Establish quantitative baselines before adversarial training.

#### Capability Baselines

| Metric | Dataset | Method |
|--------|---------|--------|
| Perplexity (general) | OpenWebText held-out (1k passages) | Cross-entropy loss, exponentiated |
| Sentiment accuracy | SST-2 validation set | Log-likelihood ranking of pos/neg continuations |
| HellaSwag | Standard benchmark | lm-evaluation-harness |
| ARC-Easy | Standard benchmark | lm-evaluation-harness |

#### Probe Detectability Baselines (at layer L)

| Probe | Architecture | Expected AUC |
|-------|-------------|-------------|
| Linear (single layer L) | LogisticRegression on h_L | > 0.80 |
| Linear (all-layer concat) | LogisticRegression on concat(h_0..h_23), dim=24576 | > 0.90 |

**Compute**: ~4 hours.

### Phase 2: Adversarial Training

#### Training Loop

```
for each outer_step (5000 total):
    # --- Inner loop: train linear probe to approximate φ* ---
    freeze θ
    for 5 inner steps:
        sample 256 labeled examples (SST-2)
        extract h_L from frozen model (detached from graph)
        update linear probe φ (dim 1024→1) to minimize BCE(probe(h_L), y_sentiment)

    # --- Outer loop: update model against probe ---
    freeze φ
    sample 16 task sequences (OpenWebText, 512 tokens)
    sample 256 labeled examples for adversarial term

    L_task   = cross_entropy(model(x_task), x_task_shifted)
    L_anchor = KL(ref_model(x_task).softmax() ‖ model(x_task).log_softmax())
    
    h_L = model(x_adv, output_hidden_states=True)[L]  # KEEP in graph
    L_probe = BCE(probe(h_L), y_sentiment)             # probe weights frozen
    
    λ_t = λ_max · min(1.0, outer_step / 1000)          # warmup over 1000 steps
    loss = L_task + α · L_anchor - λ_t · L_probe
    
    backprop loss into θ  (gradients flow: L_probe → probe(frozen) → h_L → θ)
    update θ
    
    # --- Stability safeguards ---
    if NaN detected: rollback to last checkpoint, halve λ_max
    if outer_step % 50 == 0: run 1 pure SFT step (no adversarial term)
    if outer_step % 200 == 0: reinitialize probe from scratch
    if outer_step % 100 == 0: checkpoint + quick eval (perplexity, probe AUC)
```

#### Gradient Flow Detail

The adversarial term requires gradients to flow from L_probe through the (frozen) linear probe back through h_L into θ. The probe parameters are excluded from the optimizer (or `requires_grad_(False)`) during the outer step. The activation h_L is NOT detached — it remains in the computation graph connected to θ. This is what makes the model learn to move its representations away from the linear separating hyperplane.

#### Hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| **Outer loop** | | |
| Optimizer | AdamW | Neural Chameleons |
| Learning rate | 2e-5 | Neural Chameleons (adapted for Pythia) |
| LR schedule | Cosine, 500-step warmup | Standard |
| Weight decay | 0.01 | Standard |
| Batch size (task) | 16 × 512 tokens | Neural Chameleons |
| Gradient clip | max norm 1.0 | Stability |
| α (anchor weight) | 0.1 | Neural Chameleons |
| λ_max (probe weight) | 1.0 | Tuned via ablation |
| λ warmup steps | 1000 | 20% of training |
| Total outer steps | 5000 | With early stopping |
| Precision | bfloat16 mixed | Standard |
| **Inner loop** | | |
| Optimizer | SGD | RLACE |
| Learning rate | 3e-3 | RLACE default |
| Weight decay | 1e-4 | RLACE |
| Batch size | 256 | RLACE |
| Steps per outer step | 5 | Compromise between RLACE (1) and full convergence |
| Probe architecture | Linear (1024 → 1) | F = linear classifiers |
| Probe reinit frequency | Every 200 outer steps | Prevents local minima |

#### Early Stopping Criteria

Stop training if ALL of the following hold for 500 consecutive steps:
- Probe AUC on held-out < 0.55 (near chance)
- Perplexity within 20% of baseline
- No NaN in last 200 steps

Or stop at 5000 steps regardless.

### Phase 3: Evaluation

#### 3.1 Probe Robustness (Primary Metric)

Train **fresh** linear probes (never seen during training) on the defended model's activations:

| Probe | Architecture | Success Criterion |
|-------|-------------|-------------------|
| Linear (layer L) | LogisticRegression on h_L | AUC < 0.55 |
| Linear (each layer) | LogisticRegression per layer, report best | AUC < 0.60 |
| Linear (all-layer concat) | LogisticRegression on concat(h_0..h_23) | AUC < 0.65 |

Chance performance = 0.50 AUC.

The all-layer concatenated probe tests whether the model merely displaces sentiment information to other layers rather than truly hiding it.

#### 3.2 Capability Preservation

| Metric | Success Criterion |
|--------|-------------------|
| Perplexity (general) | Within 25% of baseline |
| Sentiment accuracy (SST-2) | Within 10 points of baseline |
| HellaSwag | Within 3 points of baseline |
| ARC-Easy | Within 3 points of baseline |

#### 3.3 Stress Tests

**Concept recovery attack**: Fine-tune the defended model for 100 steps on sentiment-labeled data (lr=2e-5, batch=16). Re-evaluate probe AUC every 10 steps. Report the recovery curve (AUC vs. fine-tuning steps). If AUC recovers to > 0.80 within 100 steps, the hiding is superficial.

**Cross-dataset generalization**: Train probes on SST-2 activations, evaluate on IMDB activations. Verify the defense generalizes beyond the training distribution.

### Phase 4 (Optional): Multi-Layer Extension

If single-layer defense succeeds but all-layer concatenated linear probes still detect sentiment (per Trojan-Speak findings), extend to multi-layer:
- Select top-3 layers from Phase 0 sweep
- Train separate linear probes at each layer
- Adversarial loss: `-λ · (1/3) · Σ_k L_probe_k(φ_k*, h_Lk(θ))`
- ~3x compute cost for the adversarial term

### Future Work: Nonlinear Probe Families

Once linear evasion is established, extend F to include:
- 2-layer MLP (1024 → 512 → ReLU → 1)
- Deeper MLPs, attention probes, ensembles
- The training loop is identical — only the probe architecture changes

## Datasets

| Dataset | Source | Size | Purpose |
|---------|--------|------|---------|
| SST-2 | `glue/sst2` | 67k train / 872 val | Sentiment labels for probes |
| IMDB | `imdb` | 25k train / 25k test | Generalization eval |
| OpenWebText | `openwebtext` | 40k passages (512 tok) | L_task + L_anchor |

All splits use hash-based assignment on example ID for strict disjointness across phases.

## Ablation Schedule

| Run | λ_max | α | Inner Steps | Layer(s) | Purpose |
|-----|-------|---|-------------|----------|---------|
| 1 | 0.5 | 0.1 | 5 | Best L | Conservative baseline |
| 2 | 1.0 | 0.1 | 5 | Best L | **Default configuration** |
| 3 | 2.0 | 0.1 | 5 | Best L | Aggressive adversarial |
| 4 | 1.0 | 0.5 | 5 | Best L | Stronger anchor |
| 5 | 1.0 | 0.1 | 1 | Best L | RLACE-style single inner step |
| 6 | 1.0 | 0.1 | 20 | Best L | Near-convergence inner loop |
| 7 | 1.0 | 0.1 | 5 | Top-3 | Multi-layer defense |

**Estimated compute**: ~40–80 GPU-hours on A100-80GB for full ablation suite.

## Known Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Mode collapse (degenerate outputs) | λ warmup, SFT interleaving every 50 steps, perplexity monitoring |
| Catastrophic forgetting | KL anchor (α=0.1), early stopping on perplexity |
| NaN gradients | Detection + rollback + λ halving, gradient clipping |
| Orthogonal rotation cycling | Model rotates sentiment to a fresh orthogonal direction each time the probe updates, rather than truly erasing linear information. In 1024-d space there are many directions to cycle through. Mitigations: probe reinit exposes the model to many different linear boundaries over training so gradient signal accumulates against all directions; RLACE tracks the best probe across the entire game (non-monotonic convergence is expected). If cycling persists, consider switching to LEACE-style closed-form erasure as the inner loop (provably finds the optimal linear probe in one step, no cycling possible). |
| Probe overfitting to weak features | Probe reinit every 200 steps, fresh probes at eval |
| All-layer linear probes still detect sentiment | Multi-layer extension (Phase 4), report honestly |
| Concept recovery via minimal fine-tuning | Report recovery curve; expected limitation |

## Key References

- **RLACE** — Ravfogel et al. (ICML 2022). Minimax linear concept erasure. [`arxiv:2201.12091`](https://arxiv.org/abs/2201.12091)
- **LEACE** — Belrose et al. (NeurIPS 2023). Closed-form linear erasure. [`arxiv:2306.03819`](https://arxiv.org/abs/2306.03819)
- **Neural Chameleons** — McGuinness et al. (ICLR 2026). Finetuning to evade probes, transfer to safety monitors. [`arxiv:2512.11949`](https://arxiv.org/abs/2512.11949)
- **Obfuscated Activations** — Bailey et al. (ICLR 2026). Backdoor finetuning to bypass activation monitors. [`arxiv:2412.09565`](https://arxiv.org/abs/2412.09565)
- **Latent Adversarial Training** — Casper et al. (2024). PGD perturbations on residual stream. [`arxiv:2407.15549`](https://arxiv.org/abs/2407.15549)
- **Trojan-Speak** (2026). All-layer probes resist evasion. [`arxiv:2603.29038`](https://arxiv.org/abs/2603.29038)
- **INLP** — Ravfogel et al. (ACL 2020). Iterative null-space projection. [`arxiv:2004.07667`](https://arxiv.org/abs/2004.07667)
- **Representation Engineering** — Zou et al. (2023). Linear concept directions in activations. [`arxiv:2310.01405`](https://arxiv.org/abs/2310.01405)

## License

MIT
