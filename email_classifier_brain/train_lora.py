#!/usr/bin/env python3
"""
train_lora.py — LoRA r8 contrastive trainer on the EmbeddingGemma text tower (MPS)
==================================================================================

WI-105 P2 M7 — the retrain-orchestrator trainer artifact (design doc §5 step 2).
Runs inside the app's provisioned runtime (uv-managed CPython 3.14, torch 2.15.0
MPS from the pinned RuntimeManifest) and produces the LoRA whose human merge
checkpoint converts to the merged GGUF the §16 quality gate judges.

RUNNER CONTRACT (fixed by RetrainKit/PipelineRunner.swift, M6 — do not widen)
-----------------------------------------------------------------------------
Invoked as:

    <venv python> train_lora.py --output <runsDir>/<runID>/lora

with working directory = the SNAPSHOT brain dir, and environment:

    TRAINING_DATA_DIR=<workspace>/training_data   data-repo ROOT: *.jsonl label
                                                  files are discovered by walking
                                                  this dir recursively (train.py
                                                  semantics; labels are the
                                                  relative path minus .jsonl,
                                                  '/'-separated — the deployed
                                                  corpus yields "Jarvis/URGENT").
    PYTHONPATH=<the brain dir>                    config.py import — THE single
                                                  source of truth for input
                                                  formatting (config.format_model_input)
                                                  and the served CLS prefix.

The trainer reads NO other filesystem paths. Corpus and brain code arrive
together in the training snapshot (GET /admin/training-snapshot), so training
formats inputs with exactly the code the deployment runs — consistency by
construction. Manual-run-only flags (all defaulted from the contract) exist for
lab/development; the runner never passes them.

WHY THIS SHAPE (the validated recipe of record)
-----------------------------------------------
Every number mirrors the Oct-13/24 lab chain that produced the production
merged GGUF and validated FINDINGS §16 (dev 0.900 / test 0.889) on the served
Q8 model — reviewed against dream artifacts I-207, I-208, I-185, W-120, W-132,
W-133:

  - LoRA r8 on attention q_proj/v_proj (LIST target_modules, peft suffix
    matching), ~0.52M trainable (0.19%) — beats full-AdamW and the production
    baseline on small corpora (I-208).
  - 300 steps, batch 4 email<->class-descriptor pairs, deterministic pair
    selection (step*3 + i*7) mod n — the lab's exact arithmetic, no RNG in
    planning; contrastive cross-entropy over scale * (email_emb @ descriptor_emb^T),
    scale 20, AdamW lr 1e-4, fp32 everywhere.
  - fp32 MANDATORY (W-133: fp16 silently NaNs on this model family); NaN/inf
    probes on the loss and on produced embeddings abort loudly.
  - TRAINING TEXT = config.CLS_PROMPT + config.format_model_input(...), the
    byte-exact string production embeds (embedder._post_embeddings prepends
    CLS_PROMPT to format_model_input output; build_label_centroids.py embeds
    the same). The 1280-char body cap inside format_model_input is SIGNAL
    FILTERING, kept (I-208). Unlike the lab's stripped-"passage: " variant, we
    train on the served composition — the §16 gate measures that string
    (I-207: run the served path).
  - Deterministic seeding (default 7; recorded in meta.json): same seed + same
    corpus + same formula produce the same training trajectory, but run-to-run
    fp rounding on MPS can still flip one borderline eval row (measured
    Oct-24) — deterministic seeding is promised, bit-identical reruns are not.
  - Downstream chain (human merge checkpoint, Mode A): peft merge_and_unload
    into the text tower is EXACTLY lossless (I-207: parity cos 1.0 - 1e-8,
    Oct-24); convert_hf_to_gguf REQUIRES the sentence-transformers LEGACY
    layout — modules.json + 1_Pooling/config.json with the OLD pooling keys —
    otherwise the GGUF's pooling metadata is silently NONE and embeddings
    degrade with no error (W-132). This trainer pre-stages both files in the
    output's merge-kit/ so the human merge gets the layout right by
    construction; MERGE.md carries the full recipe of record.
  - The §16 gate itself (dev 0.900±0.01 / test >= 0.889) is enforced DOWNSTREAM
    by build_label_centroids.py against the MERGED GGUF and aborts the run.
    This trainer reports a train-fit kNN diagnostic only — an effect probe on
    the finished adapter, never a gate verdict.

FAILURE MODE
------------
Every abort prints one actionable line prefixed "TRAINING ABORT:" written for
a human in a hurry — the runner routes the stage tail into the failure the
operator sees pinned. No half-artifacts: the --output dir is populated
atomically at the end only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
import traceback

# ---------------------------------------------------------------------------
# The recipe of record (constants — change in a lab, with a FINDINGS row,
# never silently here)
# ---------------------------------------------------------------------------

MODEL_ID = "google/embeddinggemma-2"
# The base model is fetched from the hub on first run; a pinned revision must
# come from the app runtime (the merge must load the SAME revision). Optional
# env passthrough — the trainer never invents a default pin:
MODEL_REVISION_ENV = "TRAINER_BASE_MODEL_REVISION"

SEED = 7
STEPS = 300
BATCH = 4
SCALE = 20.0
LR = 1e-4
SEQ_LEN = 512          # hard cap on every tokenizer call (W-133 #4: never trust
                       # the checkpoint's junk max_seq_length — clamp yourself)
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "v_proj"]   # LIST form = peft suffix matching (validated);
                                      # a bare string would be a FULL regex
TRAINABLE_EXPECTED = (0.30e6, 2.0e6)  # abort band; the validated run was 0.52M

WORDS_PER_STEP = "step*3 + i*7"       # documented pair formula (see pair_index)

# Class descriptors of record (the lab's DESCS — keyed by the label's LAST path
# segment so deployed "Jarvis/URGENT"-style labels fold onto them).
CLASS_DESCRIPTORS = {
    "URGENT": "urgent email about an active outage, incident or failure that requires immediate action right now.",
    "FOCUS": "work email that needs my focused attention and a substantive response: reviews, proposals, planning and project tasks.",
    "REFERENCE": "informational email to keep for later reference: documentation, meeting minutes, policies, calendars and shared links.",
    "NOISE": "low-value chatty or building-wide announcement that needs no action: lunches, birthdays, casual chatter, facilities reminders.",
}


def pair_index(step: int, i: int, n: int) -> int:
    """The deterministic pair-selection formula of record (email anchors per
    step): (step*3 + i*7) mod n — the lab's exact arithmetic, no RNG."""
    return (step * 3 + i * 7) % n


# ---------------------------------------------------------------------------
# Error style: one actionable line, written for a human in a hurry
# ---------------------------------------------------------------------------

class TrainingAbort(RuntimeError):
    """Raised for every fatal trainer condition; printed as one actionable line."""


def abort(message: str, hint: str = "") -> "TrainingAbort":
    text = f"TRAINING ABORT: {message}"
    if hint:
        text += f" ACTION: {hint}"
    return TrainingAbort(text)


# ---------------------------------------------------------------------------
# W-132 merge kit: the sentence-transformers LEGACY layout the llama.cpp
# converter requires (pre-staged; the merge copies these beside the merged
# model — byte-shape of the validated Oct-24 merged dir)
# ---------------------------------------------------------------------------

ST_LEGACY_MODULES_JSON = [
    {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
    {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
    {"idx": 2, "name": "2", "path": "2_Normalize", "type": "sentence_transformers.models.Normalize"},
]

ST_LEGACY_POOLING_CONFIG = {
    "pooling_mode_mean_tokens": True,
    "pooling_mode_cls_token": False,
    "pooling_mode_lasttoken": False,
    "embedding_dimension": 768,
}


# ---------------------------------------------------------------------------
# Corpus loading — a container-test-verified mirror of the brain's
# train.load_training_data semantics (walk the data dir, labels from the
# relative path, duplicate (subject, body) resolved to the LATEST write —
# corrections MOVE, dual-labelled leftovers resolve loudly, never silently)
# ---------------------------------------------------------------------------

def load_corpus(data_dir: str) -> tuple[list[dict], list[dict]]:
    """Returns (samples, duplicate_resolutions). Sample fields mirror the
    brain's train.EmailSample (subject/body/sender/to/cc/mass_mail/
    attachment_types/label/source_path/source_line)."""
    if not os.path.isdir(data_dir):
        raise abort(
            f"TRAINING_DATA_DIR '{data_dir}' does not exist or is not a directory",
            "the snapshot unpack failed or the runner wired a wrong dir — re-run "
            "'Run now'; if it persists, check GET /admin/training-snapshot on the classifier",
        )

    samples: list[dict] = []
    file_mtimes: dict[str, float] = {}
    bad_lines = 0

    for dirpath, _, filenames in sorted(os.walk(data_dir)):
        for filename in sorted(filenames):
            if not filename.endswith(".jsonl"):
                continue
            rel_path = os.path.relpath(os.path.join(dirpath, filename), data_dir)
            label_name = os.path.splitext(rel_path)[0].replace(os.sep, "/")
            filepath = os.path.join(dirpath, filename)
            try:
                file_mtimes[filepath] = os.path.getmtime(filepath)
                with open(filepath, "r", encoding="utf-8") as f:
                    for line_no, line in enumerate(f, start=1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            bad_lines += 1
                            continue
                        samples.append({
                            "subject": entry.get("subject", ""),
                            "body": entry.get("body", ""),
                            "label": label_name,
                            "sender": entry.get("from", ""),
                            "to": entry.get("to", ""),
                            "cc": entry.get("cc", ""),
                            "mass_mail": entry.get("mass_mail", False),
                            "attachment_types": entry.get("attachment_types", []),
                            "source_path": filepath,
                            "source_line": line_no,
                        })
            except OSError as e:
                raise abort(
                    f"training data file unreadable: {filepath} ({e})",
                    "fix the snapshot dir (re-run 'Run now'); if it persists the "
                    "sandbox lost read access to a data file — report this line",
                ) from e

    if bad_lines:
        print(f"WARNING: {bad_lines} invalid JSON line(s) skipped across the corpus")

    if not samples:
        raise abort(
            f"no .jsonl training rows found under '{data_dir}'",
            "the training-data repo appears empty after the snapshot — re-run, and "
            "if it persists require a data commit with categories on the Pi",
        )

    seen: dict[tuple, tuple] = {}
    resolutions: list[dict] = []
    for sample in samples:
        key = (sample["subject"], sample["body"])
        rank = (
            file_mtimes.get(sample["source_path"], 0.0),
            sample["source_line"],
            sample["source_path"],
        )
        if key not in seen:
            seen[key] = (rank, sample)
            continue
        prev_rank, prev = seen[key]
        if rank > prev_rank:
            seen[key] = (rank, sample)
            resolutions.append({"kept": sample, "discarded": prev})
        else:
            resolutions.append({"kept": prev, "discarded": sample})

    if resolutions:
        print(f"WARNING: {len(resolutions)} duplicate (subject, body) example(s) "
              "resolved to the LATEST version (corrections MOVE, never append):")
        for res in resolutions:
            k, d = res["kept"], res["discarded"]
            print(f"  - kept {k['label']!r} ({k['source_path']}:{k['source_line']}) "
                  f"/ discarded {d['label']!r} ({d['source_path']}:{d['source_line']})")

    return [s for _, s in seen.values()], resolutions


# ---------------------------------------------------------------------------
# Input text — the SERVED composition, byte-for-byte (config.CLS_PROMPT +
# config.format_model_input(...), E5 prefix KEPT). config.py comes from the
# snapshot's brain dir: the deployment's exact formatter.
# ---------------------------------------------------------------------------

def row_to_served_text(cfg, row: dict) -> str:
    return cfg.CLS_PROMPT + cfg.format_model_input(
        subject=row["subject"],
        body=row["body"],
        sender=row["sender"],
        to=row["to"],
        cc=row["cc"],
        mass_mail=row["mass_mail"],
        attachment_types=row["attachment_types"],
    )


def descriptor_for(label: str) -> tuple[str, str]:
    """(descriptor_text, source). Lookup by the label's LAST path segment in the
    table of record; unknown labels get a deterministic auto-descriptor so a new
    category trains instead of wedging the run (curated centroids for new
    classes remain a downstream concern — surfaced in meta.json)."""
    key = label.rsplit("/", 1)[-1]
    text = CLASS_DESCRIPTORS.get(key)
    if text is not None:
        return text, "table-of-record"
    return f"email of the {key.lower()} category.", "auto-fallback"


def corpus_hash(samples: list[dict]) -> str:
    """Content hash of the loaded corpus (labels + fields, in loader order) —
    provenance the operator can diff across runs without the data dir."""
    h = hashlib.sha256()
    for s in samples:
        payload = json.dumps(
            {k: s[k] for k in ("subject", "body", "sender", "to", "cc",
                               "mass_mail", "attachment_types", "label")},
            sort_keys=True, ensure_ascii=False,
        )
        h.update(payload.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Trainer requirements — hard dependency gate BEFORE any model download
# ---------------------------------------------------------------------------

REQUIRED_MODULES = ("torch", "sentence_transformers", "transformers", "peft")


def check_requirements() -> dict:
    versions: dict[str, str] = {}
    missing: list[str] = []
    import importlib
    for mod in REQUIRED_MODULES:
        try:
            m = importlib.import_module(mod)
            versions[mod] = getattr(m, "__version__", "?")
        except Exception as e:  # ImportError and friends
            missing.append(f"{mod} ({e})")
    if missing:
        raise abort(
            "the app runtime venv is missing trainer packages: " + ", ".join(missing),
            "train_lora_requirements.txt lists the pinned set (torch comes from the "
            "RuntimeManifest wheel); refresh the runtime — a persistent miss is an "
            "install-side bug, report with this exact line",
        )
    return versions


# ---------------------------------------------------------------------------
# Contract parser (import-safe: no torch/transformers/peft at import time)
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="LoRA r8 contrastive trainer on the EmbeddingGemma text tower "
                    "(retrain-orchestrator M7; runner contract in the module docstring)")
    ap.add_argument("--output", required=True,
                    help="<runsDir>/<runID>/lora — where the adapter + merge-kit land")
    ap.add_argument("--data-dir", default=None,
                    help="overrides TRAINING_DATA_DIR (manual/lab runs only)")
    ap.add_argument("--base-model", default=MODEL_ID)
    ap.add_argument("--revision", default=None,
                    help="hub revision of the base model (else TRAINER_BASE_MODEL_REVISION)")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--allow-cpu", action="store_true", default=False,
                    help="lab/debug only: train on CPU. The pipeline requires MPS — "
                         "never wire this into the runner")
    args = ap.parse_args(argv)
    if args.steps < 1:
        raise SystemExit("--steps must be >= 1")
    if args.batch < 2:
        raise SystemExit("--batch must be >= 2 (cross-entropy needs the pair matrix)")
    return args


# ---------------------------------------------------------------------------
# Heavy path — every heavy import below happens here, never at module import
# ---------------------------------------------------------------------------

def pick_device(allow_cpu: bool) -> str:
    try:
        import torch
    except Exception as e:  # ImportError and friends
        raise abort(
            f"torch itself failed to import from the runtime venv ({e})",
            "the pinned torch wheel comes from RuntimeManifest.json via the "
            "bootstrap; use Settings > Refresh runtime — a persistent miss is an "
            "install fault, report this line",
        ) from e
    if torch.backends.mps.is_available():
        return "mps"
    if allow_cpu:
        return "cpu"
    raise abort(
        "MPS is not available — the pipeline trains on the Mac's Metal GPU only",
        "the machine needs its provisioned runtime (torch 2.15.0 MPS wheel from the "
        "pinned manifest); use Settings > Refresh runtime, and if torch reports no "
        "MPS on Apple Silicon it is an install fault — report this line. --allow-cpu "
        "exists for lab debugging and must never run unattended",
    )


def print_device_line(device: str) -> None:
    # THE device line — the only place a device is claimed; grep this in the log.
    print(f"device: {device}")


def train(data_dir: str, output_dir: str, args) -> dict:
    # Contract failures fire FIRST, on stdlib alone — before any heavy import:
    if not data_dir:
        raise abort(
            "no training data dir — TRAINING_DATA_DIR is unset and --data-dir was "
            "not given",
            "the runner always passes TRAINING_DATA_DIR; manual runs may pass "
            "--data-dir explicitly",
        )
    samples, resolutions = load_corpus(data_dir)
    labels = sorted({s["label"] for s in samples})
    counts = {lab: sum(1 for s in samples if s["label"] == lab) for lab in labels}
    content_sha = corpus_hash(samples)
    print(f"corpus: {len(samples)} rows / {len(labels)} categories: {counts}")
    print(f"corpus content sha256: {content_sha}")

    # device gate FIRST (capability signal — imports torch only): a no-MPS
    # machine aborts here before the requirements check ever runs.
    device = pick_device(args.allow_cpu)
    print_device_line(device)

    versions = check_requirements()

    import random
    import torch
    from peft import LoraConfig, get_peft_model
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    # Deterministic seeding BEFORE any model/LoRA construction (dropout draws).
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"torch {torch.__version__} | transformers {versions.get('transformers')} "
          f"| sentence-transformers {versions.get('sentence_transformers')} "
          f"| peft {versions.get('peft')}")

    # config.py must come from the SAME brain dir the trainer lives in (the
    # snapshot ships them together; `python train_lora.py` puts our dir first on
    # sys.path — this check catches a stray config from the ambient environment).
    import config as cfg
    cfg_dir = os.path.dirname(os.path.abspath(cfg.__file__))
    trainer_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.realpath(cfg_dir) != os.path.realpath(trainer_dir):
        raise abort(
            f"config.py imported from '{cfg_dir}' but the trainer lives in "
            f"'{trainer_dir}' — a foreign config could train with DIFFERENT input "
            f"formatting than the deployment runs",
            "restart the run (the runner sets the working dir + PYTHONPATH to the "
            "snapshot brain dir); never bypass this check",
        )
    print("input formatting from snapshot brain dir config.py — consistent with "
          "deployment by construction")

    descriptor_source = {lab: descriptor_for(lab)[1] for lab in labels}
    descriptor_text = {lab: descriptor_for(lab)[0] for lab in labels}
    for lab, source in descriptor_source.items():
        if source == "auto-fallback":
            print(f"WARNING: no descriptor of record for {lab!r} — using an "
                  "auto-descriptor (curate one for centroid parity)")

    # --- base model: text-only fp32 (I-185: null vision/audio configs; fp32 or
    # silent NaN — W-133) via the sentence-transformers stack — the SAME load the
    # validated merge recipe uses, so the adapter loads back bit-consistently ---
    revision = args.revision or os.environ.get(MODEL_REVISION_ENV) or None
    print(f"loading base model {args.base_model}"
          f"{f' @ revision {revision}' if revision else ''} (fp32, text-only)...")
    t0 = time.time()
    st = SentenceTransformer(
        args.base_model,
        revision=revision,
        model_kwargs={"torch_dtype": torch.float32},
        config_kwargs={"vision_config": None, "audio_config": None},
    )
    st.max_seq_length = 8192  # the checkpoint ships junk max_seq_length — clamp (W-133)
    base = st[0].auto_model
    del st
    print(f"base loaded in {time.time() - t0:.0f}s")
    base.to(device)
    base.train()

    lora = get_peft_model(base, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=list(LORA_TARGETS), bias="none",
    ))
    trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
    target_modules_seen = sorted({n.rsplit(".lora_A", 1)[0]
                                  for n, _ in lora.named_parameters() if "lora_A" in n})
    print(f"LoRA r{LORA_R} on {list(LORA_TARGETS)}: {trainable} trainable params "
          f"({trainable / 1e6:.2f}M) across {len(target_modules_seen)} modules")
    if not trainable:
        qv = sorted({n.rsplit(".lora_A", 1)[0].rsplit(".", 1)[-1]
                     for n, _ in lora.named_parameters() if "q_proj" in n or "v_proj" in n})
        print(f"module census (names matching q_proj/v_proj): {qv or 'NONE FOUND'}")
        raise abort(
            "LoRA wiring produced ZERO trainable parameters",
            "the target module names do not match this architecture — do not guess; "
            "compare the census above against the lab recipe (r8 on attention "
            "q_proj+v_proj, LIST target_modules) and report this line",
        )
    if not (TRAINABLE_EXPECTED[0] <= trainable <= TRAINABLE_EXPECTED[1]):
        raise abort(
            f"trainable params {trainable} outside the expected band "
            f"{[int(x) for x in TRAINABLE_EXPECTED]} — LoRA wired against the wrong modules",
            "expected ~0.52M for r8 on q_proj+v_proj of the validated text tower; "
            "verify the target module census above before training on",
        )

    tok = AutoTokenizer.from_pretrained(args.base_model, revision=revision)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    texts = [row_to_served_text(cfg, s) for s in samples]
    n = len(texts)
    descs = [descriptor_text[s["label"]] for s in samples]

    def encode(text_batch: list[str]) -> "torch.Tensor":
        b = tok(text_batch, padding=True, truncation=True, max_length=SEQ_LEN,
                return_tensors="pt").to(device)
        out = lora(**b)
        mask = b["attention_mask"].unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return torch.nn.functional.normalize(pooled, dim=-1)

    opt = torch.optim.AdamW([p for p in lora.parameters() if p.requires_grad],
                            lr=args.lr)

    print(f"training: {args.steps} steps x batch {args.batch} (seed {args.seed}, "
          f"pair selection {WORDS_PER_STEP} mod {n})")
    t0 = time.time()
    losses: list[float] = []
    for step in range(args.steps):
        picks = [pair_index(step, i, n) for i in range(min(args.batch, n))]
        e_emb = encode([texts[p] for p in picks])
        d_emb = encode([descs[p] for p in picks])
        logits = SCALE * (e_emb @ d_emb.T)
        loss = torch.nn.functional.cross_entropy(
            logits, torch.arange(len(picks), device=device))
        loss.backward()
        opt.step()
        opt.zero_grad()

        value = float(loss.item())
        losses.append(value)
        if not math.isfinite(value):
            raise abort(
                f"loss became non-finite ({value}) at step {step} — "
                "silent-degradation family (W-133); fp16 or a bad adapter draw "
                "corrupts the run",
                "the trainer always computes fp32; re-run once, then report with "
                "the log tail — never ship the half-trained adapter",
            )
        if step % 25 == 0 or step == args.steps - 1:
            window = losses[-25:]
            print(f"step {step:4d}/{args.steps} loss {sum(window) / len(window):.4f} "
                  f"({(time.time() - t0) / max(1, step + 1):.2f}s/step)")

    train_seconds = time.time() - t0
    print(f"training done in {train_seconds:.0f}s "
          f"(loss {sum(losses[:20]) / min(20, len(losses)):.3f} -> "
          f"{sum(losses[-20:]) / min(20, len(losses)):.3f})")

    # --- diagnostics: train-fit kNN BEFORE (adapter disabled = frozen base) and
    # AFTER, in-process — the lab's judge shape. PROXY ONLY: the real §16 gate
    # runs downstream on the merged GGUF. NaN/Inf probes are HARD aborts (never
    # hand a broken adapter to the human merge checkpoint). ---

    with torch.no_grad():
        probe_finite(_frozen_probe(lora, encode, texts))
    base_fit = _measure_train_fit(lora, encode, samples, texts, disabled=True)
    tuned_fit = _measure_train_fit(lora, encode, samples, texts, disabled=False)
    print(f"train-fit kNN (proxy diagnostic, NOT the §16 gate): "
          f"base {base_fit['accuracy']} -> tuned {tuned_fit['accuracy']} "
          f"(n={base_fit['n']})")

    regression = tuned_fit["accuracy"] < base_fit["accuracy"] - 0.05
    if regression:
        print("WARNING: tuned train-fit is BELOW base by >5pp — the §16 gate will "
              "decide quality on the merged GGUF; inspect before merging")

    adapter_state = {k: v.detach().cpu() for k, v in lora.state_dict().items()
                     if "lora" in k}
    print(f"adapter tensors captured: {len(adapter_state)} "
          f"(keys filtered on 'lora', moved to cpu)")

    return save_output(
        output_dir=output_dir,
        meta=build_meta(args, data_dir, samples, labels, counts, resolutions,
                        content_sha, trainable, target_modules_seen, versions,
                        revision, device, losses, train_seconds,
                        base_fit, tuned_fit, regression),
        adapter_state=adapter_state,
        base_model=args.base_model,
        merge_target_note="exact path is in the app's run state and merge notification",
    )


def _measure_train_fit(lora, encode, samples, texts, disabled: bool) -> dict:
    """The lab's BEFORE/AFTER judge shape: per-class centroids over train rows,
    row-by-row cosine argmax. `disabled` evaluates the frozen base through
    peft's disable_adapter(). Hidden-space, in-process — PROXY ONLY."""
    import contextlib
    import torch
    ctx = lora.disable_adapter() if disabled else contextlib.nullcontext()
    lora.eval()
    try:
        with ctx, torch.no_grad():
            per_class: dict[str, list] = {}
            eval_vecs: list = []
            for i, s in enumerate(samples):
                vec = encode([texts[i]])[0].float().tolist()
                per_class.setdefault(s["label"], []).append(vec)
                eval_vecs.append(vec)
        centroids = {}
        for lab, vecs in per_class.items():
            mean = [sum(v[k] for v in vecs) / len(vecs) for k in range(len(vecs[0]))]
            nrm = math.sqrt(sum(x * x for x in mean)) + 1e-12
            centroids[lab] = [x / nrm for x in mean]
        correct = 0
        for vec, s in zip(eval_vecs, samples):
            scores = {lab: sum(x * y for x, y in zip(vec, cent))
                      for lab, cent in centroids.items()}
            if max(scores, key=scores.get) == s["label"]:
                correct += 1
        return {"accuracy": round(correct / len(samples), 4), "n": len(samples)}
    finally:
        lora.train()


def _frozen_probe(lora, encode, texts) -> list:
    """One forward through the DISABLED adapter (frozen base) — the NaN/Inf
    effect probe runs before anything else is measured."""
    import contextlib
    lora.eval()
    try:
        with contextlib.nullcontext():
            return encode([texts[0]])[0].float().tolist()
    finally:
        lora.train()


def probe_finite(vecs: list) -> None:
    for v in vecs:
        for x in v:
            if not math.isfinite(x):
                raise abort(
                    "produced embeddings contain NaN/Inf — silent-degradation "
                    "family (W-133: fp16 quietly NaNs on this model family)",
                    "the trainer always computes fp32; re-run once, then report "
                    "with the log tail — never ship the adapter",
                )


# ---------------------------------------------------------------------------
# Output artifacts (atomic; never a half-state)
# ---------------------------------------------------------------------------

MERGE_TEMPLATE = """# Merge instructions — produced {run_stamp}

You are at the Mode A human merge checkpoint (design doc §5 step 3; P3
automates exactly this). The LoRA in this directory came from a {steps}-step
contrastive run (seed {seed}; provenance in meta.json).

THE ONLY VALIDATED CHAIN (FINDINGS §16; dream I-207: peft merge_and_unload is
exactly lossless — parity cos 1.00000000, Oct-24):

1. LOAD the base text tower EXACTLY as trained (fp32, text-only,
   revision {revision_comment}):

   from sentence_transformers import SentenceTransformer
   st = SentenceTransformer(
       "{base_model}", revision={revision_arg},
       model_kwargs={{"torch_dtype": torch.float32}},
       config_kwargs={{"vision_config": None, "audio_config": None}})

2. WRAP + LOAD this adapter (the .pt shim is the proven Oct-24 merge path;
   the LoRA wiring below must equal meta.json's lora block):

   from peft import LoraConfig, get_peft_model
   lora = get_peft_model(st[0].auto_model, LoraConfig(
       r={lora_r}, lora_alpha={lora_alpha}, lora_dropout={lora_dropout},
       target_modules={targets}, bias="none"))
   saved = torch.load("lora_adapter_final.pt", map_location="cpu")
   lora.load_state_dict(saved, strict=False)
   # expect: missing=0 unexpected=0 — anything else means the wiring or the
   # base revision moved; STOP and re-run training rather than force a merge

3. MERGE + PARITY CHECKPOINT (abort if not all-PASS — never merge on drift):

   merged = lora.merge_and_unload()
   # adapter-loaded vs merged embeddings: expect min row-cos > 0.99999,
   # max abs diff ~1e-6, identical kNN predictions (the Oct-24 numbers)

4. SAVE the merged model, then place merge-kit's files BESIDE it — the
   sentence-transformers LEGACY layout (W-132: the converter requires
   modules.json + 1_Pooling/config.json with the OLD pooling keys;
   new-style "pooling_mode": "mean" converts to a GGUF whose pooling
   metadata is silently NONE — degenerate embeddings, no error):

   merged.save_pretrained("<merged_dir>")
   cp merge-kit/modules.json <merged_dir>/
   cp -R merge-kit/1_Pooling <merged_dir>/

5. CONVERT + QUANTIZE (llama.cpp tooling; validated on nightly b11852, the app
   runtime pins b11982 — convert fresh, release-tags rot in days):

   python convert_hf_to_gguf.py <merged_dir> --outfile merged_text_f16.gguf --outtype f16
   llama-quantize merged_text_f16.gguf merged_text_q8_0.gguf Q8_0

6. PLACE the result at:

   {merge_target}

   then Run now — the runner's resume pass embeds with the app's Metal
   llama-server (--embedding --pooling mean --ubatch-size 2048) and
   build_label_centroids.py enforces the §16 gate (dev 0.900±0.01 /
   test >= 0.889), aborting the run if quality regressed.

Diagnostics recorded by the trainer are in meta.json (train-fit PROXY only —
the gate verdict belongs to the centroids build).
"""


def build_meta(args, data_dir, samples, labels, counts, resolutions, content_sha,
               trainable, target_modules_seen, versions, revision, device,
               losses, train_seconds, base_fit, tuned_fit, regression) -> dict:
    return {
        "version": 1,
        "artifact": "lora_adapter_final.pt",
        "runner_contract": {
            "output": "runs/<run_id>/lora",
            "env": {"TRAINING_DATA_DIR": "data repo root (labels from relpath)",
                    "PYTHONPATH": "snapshot brain dir (config.py import)"},
        },
        "base_model": {"id": args.base_model, "revision": revision,
                       "load": "SentenceTransformer fp32, config vision/audio = null (I-185)"},
        "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT,
                 "target_modules": list(LORA_TARGETS), "trainable_params": trainable,
                 "target_modules_seen": target_modules_seen,
                 "expected_band": [int(TRAINABLE_EXPECTED[0]), int(TRAINABLE_EXPECTED[1])]},
        "training": {"steps": args.steps, "batch": args.batch, "lr": args.lr,
                     "scale": SCALE, "seq_len": SEQ_LEN, "seed": args.seed,
                     "pair_formula": f"({WORDS_PER_STEP}) mod n",
                     "optimizer": "AdamW", "dtype": "fp32", "device": device,
                     "seconds": round(train_seconds, 1),
                     "loss_first20": round(sum(losses[:20]) / min(20, len(losses)), 4),
                     "loss_last20": round(sum(losses[-20:]) / min(20, len(losses)), 4)},
        "corpus": {"dir": data_dir,
                   "rows": len(samples), "categories": counts,
                   "content_sha256": content_sha,
                   "duplicate_resolutions": len(resolutions),
                   "descriptor_sources": {k: v for k, v in
                                          {lab: descriptor_for(lab)[1]
                                           for lab in labels}.items()
                                          if v != "table-of-record"}},
        "diagnostics": {"judge": "train-fit kNN, hidden space (PROXY — the §16 gate "
                                 "runs downstream on the merged GGUF)",
                        "base_fit": base_fit, "tuned_fit": tuned_fit,
                        "regression_flag": regression},
        "versions": versions,
        "served_text_composition": "config.CLS_PROMPT + config.format_model_input(...) "
                                   "(byte-equal to the production embed path)",
    }


def save_output(output_dir: str, meta: dict, adapter_state: dict,
                base_model: str, merge_target_note: str) -> dict:
    tmp_dir = os.path.join(output_dir, ".tmp")

    if os.path.lexists(output_dir):
        if os.path.isdir(output_dir) and not os.listdir(output_dir):
            shutil.rmtree(output_dir)
        else:
            raise abort(
                f"output directory '{output_dir}' already exists and is not empty",
                "the runner allocates a fresh runs/<run_id>/lora per run — a stale "
                "non-empty dir means a previous run died mid-save; delete it or "
                "pass a different --output",
            )

    try:
        os.makedirs(tmp_dir, exist_ok=True)

        # 1. the adapter of record — the state-dict shim the validated merge
        # loads (torch.save, 'lora'-keyed, ALWAYS a cpu copy)
        import torch
        torch.save(adapter_state, os.path.join(tmp_dir, "lora_adapter_final.pt"))

        # 2. merge-kit — the W-132 legacy layout, pre-staged
        kit_dir = os.path.join(tmp_dir, "merge-kit", "1_Pooling")
        os.makedirs(kit_dir, exist_ok=True)
        with open(os.path.join(tmp_dir, "merge-kit", "modules.json"), "w") as f:
            json.dump(ST_LEGACY_MODULES_JSON, f, indent=1)
        with open(os.path.join(kit_dir, "config.json"), "w") as f:
            json.dump(ST_LEGACY_POOLING_CONFIG, f, indent=1)

        # 3. meta.json — provenance + diagnostics
        with open(os.path.join(tmp_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1, default=str)

        # 4. MERGE.md — the human checkpoint's recipe of record
        lora_meta = meta["lora"]
        revision = meta["base_model"]["revision"]
        merge_md = MERGE_TEMPLATE.format(
            run_stamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            steps=meta["training"]["steps"], seed=meta["training"]["seed"],
            base_model=base_model,
            revision_comment=(f"pinned {revision}" if revision
                              else "NOT PINNED — pin the revision you train-verified "
                                   "against (meta.json base_model.revision is null)"),
            revision_arg=(f'"{revision}"' if revision else "None"),
            lora_r=lora_meta["r"], lora_alpha=lora_meta["alpha"],
            lora_dropout=lora_meta["dropout"],
            targets=json.dumps(lora_meta["target_modules"]),
            merge_target=("runs/<run_id>/merged_text_q8_0.gguf — " + merge_target_note),
        )
        with open(os.path.join(tmp_dir, "MERGE.md"), "w") as f:
            f.write(merge_md)
    except TrainingAbort:
        _discard_tmp(tmp_dir)
        raise
    except Exception as e:
        _discard_tmp(tmp_dir)
        raise abort(f"artifact serialization failed: {e}",
                    "re-run once; a repeated failure at the same step is a bug — "
                    "report with this line") from e

    # atomic flip: tmp contents staged, then moved in one pass
    os.makedirs(output_dir, exist_ok=True)
    for name in sorted(os.listdir(tmp_dir)):
        shutil.move(os.path.join(tmp_dir, name), os.path.join(output_dir, name))
    os.rmdir(tmp_dir)
    return {"output": output_dir, "files": sorted(os.listdir(output_dir))}


def _discard_tmp(tmp_dir: str) -> None:
    shutil.rmtree(tmp_dir, ignore_errors=True)


def main(argv=None) -> int:
    try:
        args = parse_args(argv)
        data_dir = args.data_dir or os.environ.get("TRAINING_DATA_DIR", "")
        train(data_dir, args.output, args)
    except TrainingAbort as e:
        print(str(e), file=sys.stderr)
        return 1
    except SystemExit:
        raise
    except Exception as e:
        traceback.print_exc()
        print(abort(f"unexpected failure: {e}",
                    "re-run once; a repeated failure at the same step is a bug — "
                    "report this line and the log tail"), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
