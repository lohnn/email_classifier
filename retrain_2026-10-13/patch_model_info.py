#!/usr/bin/env python3
"""Post-train provenance patch for MODEL_INFO.json.

train.py's _git_info(TRAINING_DATA_DIR) runs on the train-shards dir (not a
git repo) -> training_data_commit comes out empty. This script patches the
real provenance in: data-repo source commit, dedup/split manifest summary,
micro-detail the coordinator asked for (created-by stack).
"""
import json, os, sys

WORK = os.path.dirname(os.path.abspath(__file__))
info_path = os.path.join(WORK, "model", "MODEL_INFO.json")
manifest = json.load(open(f"{WORK}/corpus/manifest.json"))

info = json.load(open(info_path))
info["training_data_repo"] = "https://github.com/lohnn/email_classifier_data"
info["training_data_commit"] = manifest["data_repo_head"]
info["training_data_dedup"] = {
    "gross_records": manifest["gross_records"],
    "dual_labeled_groups_dropped": manifest["dual_labeled_groups_dropped"],
    "net_records": manifest["net_records"],
    "split": manifest["splits"],
    "splits_per_class": manifest["splits_per_class"],
    "seed": manifest["seed"],
    "protocol": manifest["protocol"],
}
info["created_by"] = {"setfit": "1.2.0", "sentence_transformers": "5.7.0",
                      "sklearn": "1.9.1", "torch": "2.14.1",
                      "note": "Pi runtime is setfit 1.1.0/ST 3.4.1/torch 2.3.0 — verify load on Pi before relying (see UPLOAD-NOTES.md)"}
info["retrain_reason"] = ("first fresh-data retrain; deployed Pi model was trained on the Feb-era corpus "
                          "(labels frozen 2026-03-26 .. 2026-10-12, 3 auto-commits Oct 12-13 landed)")
with open(info_path, "w") as f:
    json.dump(info, f, indent=2, ensure_ascii=False)
print("patched", info_path, "->", info["training_data_commit"][:12])
