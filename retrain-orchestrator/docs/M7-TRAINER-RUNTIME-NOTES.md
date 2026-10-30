# M7 — trainer runtime validation notes (operator round)

`email_classifier_brain/train_lora.py` (WI-105 P2 M7) is the LoRA-on-MPS
trainer. This file tells the operator what a validation round looks like on
the Mac. Container-side truth (22 pytest, incl. loader parity vs `train.py`
and the W-132 merge-kit layout) is already green; MPS training itself can only
be validated here.

## The contract it must run under (fixed, M6)

Runner invocation (PipelineRunner.swift, training stage):

```
<venv python> train_lora.py --output <runsDir>/<runID>/lora
# working dir: the SNAPSHOT brain dir
# env: TRAINING_DATA_DIR=<workspace>/training_data (data repo root)
#      PYTHONPATH=<snapshot brain dir>
```

Manual/lab validation is the same shape, with explicit `--data-dir`.

## Tier 1 — recipe validation (runnable today, no app changes)

Any Python with the pinned stack (`email_classifier_brain/train_lora_requirements.txt`;
the lab training venv already carries exactly that set + torch) on the Mac:

```
cd <brain dir>
python3 train_lora.py --output /tmp/m7_lora \
    --data-dir /workspace/projects/email_classifier_data   # or the real data dir
```

Expect in the log, in order:

1. `corpus: N rows / 4 categories` (+ content sha; the live repo also prints
   the 4 known duplicate resolutions — corrections MOVE, resolved by mtime)
2. `device: mps`            ← THE device line; a run without it proves nothing
3. `LoRA r8 on ['q_proj', 'v_proj']: ~520k trainable params (0.52M)...`
   (outside the [300k, 2M] band => the run aborts loudly — right behavior)
4. step lines every 25 steps, loss trending down (first20 -> last20 in meta.json)
5. `train-fit kNN (proxy diagnostic, NOT the §16 gate): base -> tuned ...`
6. artifacts: `lora/`: `lora_adapter_final.pt`, `meta.json`, `MERGE.md`,
   `merge-kit/{modules.json, 1_Pooling/config.json}`

Paste back: the full console output (or its tail), the `device:` line, the
LoRA param line, the diagnostics line, plus `ls -R /tmp/m7_lora` and the
`meta.json` contents. Failures verbatim — `TRAINING ABORT:` lines are written
for exactly that.

## Tier 2 — provisioned-runtime validation (after venv wiring lands)

Same command but with the app's own runtime python:

```
<containerDir>/rt/venv/bin/python train_lora.py --output /tmp/m7_lora_t2 \
    --data-dir <TrainingData root same as the classifier's>
```

This tier only means something once the bootstrap installs
`train_lora_requirements.txt` into the app venv (the M5-runtime-side wiring is
an open integration item — see the M7 report; torch stays manifest-pinned).
Missing packages abort with the exact missing-module list — that text is the
diagnostic.

## What NOT to conclude from Tier 1/2

- Train-fit kNN numbers are PROXIES (in-process, hidden space). The real gate
  is §16 (dev 0.900±0.01 / test >= 0.889), enforced downstream by
  `build_label_centroids.py` against the MERGED GGUF through the served Metal
  llama-server. It aborts the run if quality regressed.
- The pair-selection formula, LoRA targets, steps/scale/lr are the
  Oct-13/24-validated recipe of record — don't tune them inside this file
  without a lab run + a FINDINGS row.
- Deterministic seeding (default 7) is promised; bit-identical reruns are not
  (MPS fp rounding can flip one borderline row — measured Oct-24).
