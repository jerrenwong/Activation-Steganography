# Project: Activation Steganography

## What this is
Minimax training to hide **sentiment** from **linear probes** in **Pythia-410M** activations while preserving model capabilities. Proof-of-concept for whether models can evade activation-based safety monitors.

## Formulation
```
min_θ [ L_task(θ) + α·L_anchor(θ,θ_ref) - λ·L_probe(φ*, h_l(θ)) ]
where φ* = argmin_φ L_probe(φ, h_l(θ))    (best probe from family F)
```
- F = linear classifiers for now (pluggable to MLPs later)
- Inner loop: SGD lr=3e-3, 5 steps, trains linear probe on detached activations
- Outer loop: AdamW lr=2e-5, updates model to fool probe while preserving capability
- λ warmup 0→1.0 over 1000 steps, α=0.1 (KL anchor)

## Current state
All code implemented. Not yet run. Pipeline: `python -m scripts.run_all`
- Baselines: per-layer probe sweep on SST-2 → picks target layer L
- Training: adversarial loop, 5000 steps, logs fresh probe AUC + generation samples every 100 steps
- Evaluation: fresh per-layer probes, IMDB cross-dataset, concept recovery attack

## Key design decisions made
- **Model**: Pythia-410M (24 layers, 1024 hidden dim, ~8-10GB VRAM). English-only is fine for sentiment.
- **Concept**: Sentiment (positive/negative via SST-2). Chosen as clean proxy for safety-relevant concepts.
- **Probe family**: Linear only for now. Nonlinear (MLP) probes deferred to future work.
- **No perplexity tracking**: Removed to simplify. Generation logging serves as capability monitor instead.
- **No all-layer concat probe**: Removed from eval for now. Known from Trojan-Speak that these are very hard to fool with single-layer defense.
- **Known risk**: Orthogonal rotation cycling — model may just rotate sentiment to fresh directions rather than truly erasing linear information. Probe reinit every 200 steps is the mitigation.

## Key files
- `configs/default.yaml` — all hyperparameters
- `src/training/adversarial_trainer.py` — the core training loop
- `src/probes/probe.py` — LinearProbe (differentiable, for training) + SklearnProbe (for eval)
- `scripts/run_all.py` — end-to-end overnight script

## Running the experiments overnight

### Quick start
```bash
cd /workspace/Activation-Steganography
bash start.sh
```
This installs deps, verifies GPU, and launches the full pipeline in background via `nohup`.

### What the pipeline does (scripts/run_all.py)
1. **Phase 0 — Probe sweep** (~2h): Extracts activations at all 24 layers on SST-2, trains sklearn LogisticRegression per layer, picks target layer L (highest val AUC). Saves `results/baselines.json`.
2. **Phase 2 — Adversarial training** (~4-6h): 5000 outer steps of minimax training. Inner loop trains linear probe on detached activations (5 SGD steps). Outer loop updates model with `loss = L_task + α·L_anchor - λ·L_probe`. Logs fresh probe AUC + generation samples every 100 steps. Checkpoints to `checkpoints/step_*/`. Saves `results/training_metrics.json`, `results/training_eval.json`, `results/training_generations.json`.
3. **Phase 3 — Evaluation** (~1h): Fresh per-layer probe sweep on defended model, IMDB cross-dataset generalization, concept recovery attack (100 fine-tuning steps). Saves `results/eval.json`.

### Monitoring
```bash
tail -f results/run_all.log    # structured experiment logs
tail -f results/stdout.log     # raw output
kill -0 $(cat results/run.pid) 2>/dev/null && echo 'Running' || echo 'Done'
```

### Results files
- `results/baselines.json` — per-layer probe AUCs + selected target layer
- `results/training_metrics.json` — per-step loss curves (l_task, l_anchor, l_probe, lambda)
- `results/training_eval.json` — fresh probe AUC at every 100 steps during training
- `results/training_generations.json` — model completions on fixed prompts at every 100 steps
- `results/eval.json` — final: target layer AUC, best layer AUC, IMDB cross-dataset AUC, recovery curve

### Success criteria
- Target layer probe AUC: baseline >0.80 → defended <0.55
- Best layer post-defense AUC: <0.60
- IMDB cross-dataset AUC: similar to SST-2 (defense generalizes)
- Recovery attack: if AUC recovers to >0.80 within 100 steps, hiding is superficial

### Wandb
Disabled by default. To enable: `export WANDB_API_KEY=your_key` before running `start.sh`.

### GPU requirements
~8-10GB VRAM (Pythia-410M params + ref model + optimizer + activations). Runs on A100, RTX 3090/4090/5090.

## Key references
- Neural Chameleons (ICLR 2026) — most directly relevant, lr=2e-5, α=0.1
- RLACE (ICML 2022) — minimax template, SGD lr=0.003
- Trojan-Speak (2026) — all-layer probes resist evasion (AUC>0.97)
- LEACE (NeurIPS 2023) — closed-form linear erasure, fallback if cycling occurs
