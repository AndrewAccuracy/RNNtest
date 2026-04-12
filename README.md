# Delayed Memory Experiment For RNN PPT

This project supports an RNN presentation in **two phases**:

1. **Phase 1 — Why recurrence (vs older baselines)**  
   Compare a fixed-window model and a **full-sequence flattened MLP** against an **RNN**. The fixed-window baseline often cannot see the label-relevant token; the flat MLP sees the **same tokens** as the RNN but without weight sharing over time—useful for contrasting *inductive bias* and scaling.

2. **Phase 2 — Why LSTM (vs plain RNN)**  
   Compare **RNN** and **LSTM** when dependencies get long: plain RNNs often become unstable; LSTMs are often more stable in the same regime.

## Task

Each sequence looks like this:

```text
A B c d e f g ... ?
A A c d e f g ... ?
B A c d e f g ... ?
...
```

Meaning:

- in Phase 2, the first `k` tokens form a pattern
- the middle tokens are random distractors
- the last token (`?`) asks the model to predict the pattern class

With `k=2`, there are 4 pattern classes. This is more informative than a single-token memory task and better exposes the transition from short dependency to long dependency.

## Compared Models

- `History-MLP`: only sees the last few tokens (fixed window), representing a fixed-window historical model
- `Flat-MLP`: full sequence, one-hot flattened then MLP (no recurrence; input size grows with sequence length)
- `RNN`: reads the whole sequence and propagates hidden states
- `LSTM`: reads the whole sequence with gated memory

## Additional Metrics

Besides `accuracy vs gap`, the project now also computes:

- `critical gap`: the first dependency length where mean accuracy drops below a chosen threshold
- `seed stability`: variability across multiple random seeds
- `training time to threshold`: how many epochs are needed before training loss drops below a chosen threshold
- `matched-parameter comparison`: an approximate parameter-matched RNN/LSTM comparison table

## Run

Activate the local project environment:

```bash
source .venv/bin/activate
```

Run the default experiment:

```bash
python experiment.py
```

Optional custom run:

```bash
python experiment.py --epochs 20 --hidden-size 16 --gaps 5 10 18 20 22 26 30 50 100 --seeds 1 2 3 4 5 --phase2-prefix-len 2 --match-lstm-params
```

**Recommended for slides (full gap sweep + 5 seeds):**

```bash
python experiment.py --preset slides
```

**Quick smoke test (short gaps, one seed):**

```bash
python experiment.py --preset quick --epochs 2
```

### Experiment 2 Final Copy Task

For the cleaner `RNN vs LSTM` comparison, use the dedicated copy-memory script. The current best showcase setup is:

```bash
python experiment_lstm_copy_task.py \
  --gaps 15 18 \
  --seeds 1 2 3 \
  --memory-length 3 \
  --train-samples 1800 \
  --test-samples 400 \
  --batch-size 64 \
  --epochs 60 \
  --embedding-dim 32 \
  --hidden-size 128 \
  --num-symbols 4 \
  --token-threshold 0.75 \
  --output-dir outputs_exp2_copy_task_final
```

This run produces a cleaner “why LSTM” story:

- `RNN` stays near chance on delayed copy recall
- `LSTM` reaches much higher token recall and sequence recall
- the learning-curve figures make the convergence gap easier to explain in a PPT

If you only use small gaps (e.g. 5 and 10), RNN and LSTM will often both reach ~100% accuracy and the Phase 2 curves overlap—this is expected, not a bug. Use longer gaps (included in `--preset slides`) to see separation.

### Export data to disk (optional)

Write JSONL train/test splits under `data/delayed_memory/gap_XXX/` using the same RNG as in-code generation (train: `first_seed + 100`, test: `first_seed + 200`):

```bash
python experiment.py --export-data
```

Custom root:

```bash
python experiment.py --export-data --data-export-root data/delayed_memory
```

### Train from exported JSONL (optional)

When set, every seed uses the **same** fixed train/test files; variance across seeds reflects initialization and batch order, not resampled data:

```bash
python experiment.py --data-dir data/delayed_memory
```

## Outputs

The script writes results into `outputs/`.

**Phase-focused figures (recommended for slides):**

- `phase1_accuracy_vs_gap_bands.png`, `phase1_accuracy_heatmap.png`, `phase1_seed_accuracy_scatter.png`, `phase1_training_loss_gap_max.png`
- `phase2_accuracy_vs_gap_bands.png`, `phase2_accuracy_heatmap.png`, `phase2_seed_accuracy_scatter.png`, `phase2_training_loss_gap_max.png`, `phase2_time_to_threshold.png`

**All four models together:**

- `accuracy_vs_gap_bands.png`, `accuracy_heatmap.png`, `seed_accuracy_scatter.png`, `training_loss_gap_max.png`

**Metrics:**

- `phase1_metrics.json`, `phase2_metrics.json`, `metrics.json` (all models), `metrics_combined.json` (nested `phase1` / `phase2` / `all_models`)
- `phase1_summary.txt`, `phase2_summary.txt`, `summary.txt`
- `matched_parameter_table.md`

## Suggested PPT Usage

- **Phase 1 slide:** `phase1_accuracy_vs_gap_bands.png` (optionally add `phase1_accuracy_heatmap.png`).
- **Phase 2 slide:** `phase2_accuracy_vs_gap_bands.png` (optionally add `phase2_seed_accuracy_scatter.png` to show stability).

If you need one single overview chart with every model, use `accuracy_vs_gap_bands.png`.

For the dedicated copy-memory experiment, see `outputs_exp2_copy_task_final/`:

- `exp2_copy_token_accuracy.png`
- `exp2_copy_sequence_accuracy.png`
- `exp2_copy_token_learning_curve.png`
- `exp2_copy_sequence_learning_curve.png`
- `exp2_copy_summary.txt`
