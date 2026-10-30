"""
Tests for train_lora.py — WI-105 P2 M7 (retrain-orchestrator trainer).

Container-honest scope: every CONTRACT surface (arg/env handling, corpus
shaping vs the brain's train.py reference loader, served-text composition,
W-132 merge-kit layout, atomic save shape, import-time dependency safety).
The MPS training loop itself CANNOT run here (no Metal in-container) — its
validation is operator-side in the provisioned runtime by explicit agreement;
these tests pin everything that CAN be honest.
"""

import importlib
import json
import os
import subprocess
import sys

import pytest

BRAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BRAIN_DIR)

TORCH = pytest.importorskip("torch")  # the container venv carries cpu torch

train_lora = importlib.import_module("train_lora")


# ---------------------------------------------------------------------------
# runner contract: args + env
# ---------------------------------------------------------------------------

def test_parse_args_defaults_match_recipe_of_record():
    args = train_lora.parse_args(["--output", "/tmp/lora"])
    assert args.output == "/tmp/lora"
    assert args.base_model == "google/embeddinggemma-2"
    assert args.steps == 300
    assert args.batch == 4
    assert args.lr == pytest.approx(1e-4)
    assert args.seed == 7
    assert args.allow_cpu is False
    assert args.revision is None


def test_parse_args_requires_output():
    with pytest.raises(SystemExit):
        train_lora.parse_args([])  # --output is the runner's one fixed arg


def test_pair_formula_is_the_lab_arithmetic():
    # (step*3 + i*7) % n — deterministic, no RNG; hand-checked lab values
    assert [train_lora.pair_index(s, 0, 45) for s in range(5)] == [0, 3, 6, 9, 12]
    assert [train_lora.pair_index(10, i, 45) for i in range(4)] == [30, 37, 44, 6]
    # n==0 is structurally unreachable: load_corpus aborts an empty corpus
    # with an actionable error BEFORE any pair math runs.


# ---------------------------------------------------------------------------
# env contract: TRAINING_DATA_DIR
# ---------------------------------------------------------------------------

def test_missing_data_dir_is_an_actionable_abort():
    with pytest.raises(train_lora.TrainingAbort) as exc:
        train_lora.load_corpus("/nonexistent/training_data_dir")
    msg = str(exc.value)
    assert msg.startswith("TRAINING ABORT:")
    assert "TRAINING_DATA_DIR" in msg
    assert "/nonexistent/training_data_dir" in msg
    assert "ACTION:" in msg  # the failure becomes the operator's pinned notification


def test_nth_empty_dir_is_an_actionable_abort(tmp_path):
    empty = tmp_path / "training_data"
    empty.mkdir()
    with pytest.raises(train_lora.TrainingAbort) as exc:
        train_lora.load_corpus(str(empty))
    assert "no .jsonl training rows" in str(exc.value)


def test_missing_data_dir_fires_before_any_heavy_import(tmp_path, monkeypatch):
    # Order promise of train(): the env contract aborts on stdlib alone —
    # before torch import (which would fail on machines without the runtime).
    monkeypatch.delenv("TRAINING_DATA_DIR", raising=False)
    args = train_lora.parse_args(["--output", str(tmp_path / "lora")])
    with pytest.raises(train_lora.TrainingAbort) as exc:
        train_lora.train("", args.output, args)
    msg = str(exc.value)
    assert "no training data dir" in msg and "TRAINING_DATA_DIR" in msg
    assert "MPS" not in msg  # must not surface the device error for an env fault


def test_device_gate_requires_mps_outside_lab_mode(tmp_path):
    # With a real corpus but no Metal (this container), the only honest outcome
    # is the MPS abort — never a silent CPU run (the runner never passes
    # --allow-cpu).
    data = tmp_path / "training_data"
    (data / "Jarvis").mkdir(parents=True)
    _write_jsonl(str(data / "Jarvis" / "URGENT.jsonl"),
                 [_mk_row("s", "b"), _mk_row("s2", "b2")])
    args = train_lora.parse_args(["--output", str(tmp_path / "lora")])
    with pytest.raises(train_lora.TrainingAbort) as exc:
        train_lora.train(str(data), args.output, args)
    assert "MPS is not available" in str(exc.value)


# ---------------------------------------------------------------------------
# corpus shaping: full parity with the brain's train.py reference loader
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def brain_train_module():
    pytest.importorskip("setfit")  # the reference loader drags the SetFit stack in
    brain_train = importlib.import_module("train")
    return brain_train


def _write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _mk_row(subject, body, frm="sender@example.com", to="me@example.com",
            cc="", mass_mail=False, attachment_types=None):
    # JSONL shape ("from") plus the loader-normalized alias ("sender") — the
    # fixture works both as corpus files (loader reads "from") and as an
    # already-loaded sample (row_to_served_text reads "sender").
    return {"subject": subject, "body": body, "from": frm, "sender": frm,
            "to": to, "cc": cc, "mass_mail": mass_mail,
            "attachment_types": attachment_types or []}


def test_corpus_loader_matches_train_py_reference(tmp_path, brain_train_module):
    # Same fixtures through BOTH loaders must produce the same samples:
    # labels from relative paths, categories auto-discovered, duplicates
    # resolved to the LATEST mtime (corrections MOVE, never append).
    data = tmp_path / "training_data"
    legacy_file = data / "Jarvis" / "URGENT.jsonl"
    _write_jsonl(legacy_file, [
        _mk_row("server down", "the db is on fire, please help"),
        _mk_row("pager", "all hands on the incident channel"),
    ])
    noise_file = data / "Jarvis" / "NOISE.jsonl"
    _write_jsonl(noise_file, [
        _mk_row("lunch", "pizza at 12 in the kitchen"),
        _mk_row("server down", "the db is on fire, please help"),  # the MOVE-era dup
    ])
    # control the duplicate resolution: URGENT written EARLIER in wall time, so
    # the NOISE copy "wins" — then flip mtimes to prove mtime is the rank
    now = 1_700_000_000.0
    os.utime(noise_file, (now, now))
    os.utime(legacy_file, (now + 60, now + 60))

    mine, resolutions = train_lora.load_corpus(str(data))
    theirs = brain_train_module.load_training_data(str(data))

    key = lambda s: (s["subject"], s["body"], s["label"])
    assert sorted(map(key, mine)) == sorted(
        (s.subject, s.body, s.label) for s in theirs)
    # exactly one duplicate resolution happened, and it kept the LATEST file
    assert len(resolutions) == 1
    kept = resolutions[0]["kept"]
    assert kept["label"] == "Jarvis/URGENT"  # URGENT.jsonl has the LATER mtime


def test_corpus_shapes_and_field_mapping(tmp_path):
    data = tmp_path / "training_data"
    _write_jsonl(data / "Jarvis" / "FOCUS.jsonl", [
        _mk_row("project review", "please review the proposal",
                attachment_types=["PDF"], mass_mail=False,
                frm="a@b.com", to="me@c.com", cc="d@e.com"),
    ])
    samples, resolutions = train_lora.load_corpus(str(data))
    assert resolutions == []
    s = samples[0]
    assert s["label"] == "Jarvis/FOCUS"
    assert s["sender"] == "a@b.com"          # jsonl 'from' -> sender
    assert s["attachment_types"] == ["PDF"]
    assert s["mass_mail"] is False
    assert s["source_line"] == 1


# ---------------------------------------------------------------------------
# served-text composition: byte-equal to what production embeds
# ---------------------------------------------------------------------------

def test_served_text_is_cls_prompt_plus_format_model_input_prefix_kept():
    import config
    row = _mk_row("invoice", "here is your receipt for the workshop",
                  attachment_types=["PDF"], frm="b@c.com")
    mine = train_lora.row_to_served_text(config, row)
    theirs = config.CLS_PROMPT + config.format_model_input(
        subject=row["subject"], body=row["body"], sender=row["sender"],
        to=row["to"], cc=row["cc"], mass_mail=row["mass_mail"],
        attachment_types=row["attachment_types"])
    assert mine == theirs
    # served parity: the E5 "passage: " prefix is KEPT inside the CLS-prefixed
    # string (embedder._post_embeddings composes CLS_PROMPT + format_model_input)
    assert mine.startswith("task: classification | query: ")
    assert config.E5_PREFIX in mine


def test_cls_prompt_is_the_byte_string_of_record():
    import config
    assert config.CLS_PROMPT == "task: classification | query: "


def test_body_cap_survives_into_served_text():
    import config
    row = _mk_row("long body", "x" * 5000)
    text = train_lora.row_to_served_text(config, row)
    assert len(row["body"]) > config.MAX_BODY_CHARS
    assert config.TRUNCATION_MARKER in text  # the 1280-char cap is signal, kept


# ---------------------------------------------------------------------------
# descriptors
# ---------------------------------------------------------------------------

def test_descriptor_folding_over_hierarchical_labels():
    text, source = train_lora.descriptor_for("Jarvis/URGENT")
    assert source == "table-of-record"
    assert text == train_lora.CLASS_DESCRIPTORS["URGENT"]


def test_descriptor_auto_fallback_is_deterministic():
    a = train_lora.descriptor_for("Jarvis/NEWTHING")
    b = train_lora.descriptor_for("OTHER/NEWTHING")
    assert a == b  # folded onto the same suffix
    assert a[1] == "auto-fallback"
    assert "newthing" in a[0]


def test_corpus_hash_is_content_stable():
    base = _mk_row("s", "b")
    rows = [dict(base, label="L")]
    rows2 = [dict(_mk_row("s", "b"), label="L")]
    assert train_lora.corpus_hash(rows) == train_lora.corpus_hash(rows2)
    rows3 = [dict(_mk_row("s", "b"), label="Z")]
    assert train_lora.corpus_hash(rows) != train_lora.corpus_hash(rows3)


# ---------------------------------------------------------------------------
# W-132: the merge-kit must pin the sentence-transformers LEGACY layout
# ---------------------------------------------------------------------------

def test_merge_kit_modules_json_is_legacy_three_module_layout():
    mods = train_lora.ST_LEGACY_MODULES_JSON
    assert [m["path"] for m in mods] == ["", "1_Pooling", "2_Normalize"]
    assert "Pooling" in json.dumps(mods) and "Normalize" in json.dumps(mods)


def test_merge_kit_pooling_config_uses_old_keys_only():
    cfg = train_lora.ST_LEGACY_POOLING_CONFIG
    assert cfg["pooling_mode_mean_tokens"] is True      # the OLD key form
    assert "pooling_mode" not in cfg                     # new-style key is poison
    assert cfg["embedding_dimension"] == 768


def test_merge_instructions_carry_the_validated_chain():
    md = train_lora.MERGE_TEMPLATE
    assert "merge_and_unload" in md                        # peft merge of record
    assert "convert_hf_to_gguf.py" in md                   # llama.cpp converter
    assert "llama-quantize" in md and "Q8_0" in md         # quantization step
    assert "--pooling mean" in md and "--ubatch-size 2048" in md
    assert "1_Pooling" in md and "modules.json" in md      # legacy layout copy step


# ---------------------------------------------------------------------------
# atomic save shape
# ---------------------------------------------------------------------------

def _fake_meta():
    return {
        "base_model": {"id": "google/embeddinggemma-2", "revision": "abc123"},
        "lora": {"r": 8, "alpha": 16, "dropout": 0.05,
                 "target_modules": ["q_proj", "v_proj"]},
        "training": {"steps": 300, "batch": 4, "lr": 1e-4, "scale": 20.0,
                     "seq_len": 512, "seed": 7, "pair_formula": "(step*3 + i*7) mod n",
                     "optimizer": "AdamW", "dtype": "fp32", "device": "mps",
                     "seconds": 42.0, "loss_first20": 2.0, "loss_last20": 0.5},
        "corpus": {"rows": 4, "categories": {"Jarvis/URGENT": 4},
                   "content_sha256": "deadbeef", "duplicate_resolutions": 0,
                   "descriptor_sources": {}},
        "diagnostics": {"judge": "proxy", "base_fit": {"accuracy": 0.5, "n": 4},
                        "tuned_fit": {"accuracy": 0.75, "n": 4},
                        "regression_flag": False},
        "versions": {"torch": "test"},
    }


def test_save_output_atomic_layout_complete(tmp_path):
    out = tmp_path / "lora"
    result = train_lora.save_output(
        str(out), _fake_meta(), {"lora_A.default.weight": [0.0]},
        base_model="google/embeddinggemma-2", merge_target_note="note")
    assert result["output"] == str(out)
    names = set(result["files"])
    assert {"lora_adapter_final.pt", "meta.json", "MERGE.md",
            "merge-kit"} <= names
    assert json.loads((out / "merge-kit" / "modules.json").read_text()) \
        == train_lora.ST_LEGACY_MODULES_JSON
    pooling = json.loads((out / "merge-kit" / "1_Pooling" / "config.json").read_text())
    assert pooling["pooling_mode_mean_tokens"] is True
    meta = json.loads((out / "meta.json").read_text())
    assert meta["corpus"]["rows"] == 4
    assert meta["lora"]["target_modules"] == ["q_proj", "v_proj"]
    assert not (out / ".tmp").exists()  # the staging dir never leaks


def test_save_output_refuses_existing_nonempty_dir(tmp_path):
    out = tmp_path / "lora"
    out.mkdir()
    (out / "stale-artifact.bin").write_bytes(b"x")
    with pytest.raises(train_lora.TrainingAbort) as exc:
        train_lora.save_output(str(out), _fake_meta(), {}, "m", "note")
    assert "already exists" in str(exc.value)


def test_save_output_allows_precreated_empty_dir(tmp_path):
    out = tmp_path / "lora"
    out.mkdir()
    result = train_lora.save_output(str(out), _fake_meta(),
                                    {"k": [0.0]}, base_model="m",
                                    merge_target_note="note")
    assert "MERGE.md" in result["files"]  # empty slot == treat as fresh


# ---------------------------------------------------------------------------
# import-time dependency safety (the M6 runner starts this file with the
# app's python; contract surfaces must load without torch/transformers/peft)
# ---------------------------------------------------------------------------

def test_import_time_safety_blocks_heavy_modules():
    # The runner starts this file cold; the CONTRACT surface (import + parse)
    # must load even with torch/transformers/peft unavailable (import-time
    # safety) — a MetaPathFinder import-blocker proves it in a subprocess.
    code = (
        "import sys, importlib.abc\n"
        "BLOCKED = ['torch', 'transformers', 'peft', 'sentence_transformers']\n"
        "class _Block(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, *a, **k):\n"
        "        if name in BLOCKED:\n"
        "            raise ImportError('blocked for import-time test: ' + name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Block())\n"
        f"sys.path.insert(0, {BRAIN_DIR!r})\n"
        "import train_lora\n"
        "args = train_lora.parse_args(['--output', 'x'])\n"
        "assert args.steps == 300 and args.batch == 4\n"
        "assert train_lora.pair_index(10, 2, 45) == 44\n"
        "print('CONTRACT-SURFACE-IMPORT-OK')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, cwd=BRAIN_DIR)
    assert proc.returncode == 0, proc.stderr
    assert "CONTRACT-SURFACE-IMPORT-OK" in proc.stdout, proc.stderr
