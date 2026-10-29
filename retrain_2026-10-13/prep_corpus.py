#!/usr/bin/env python3
"""Prep the REAL corpus for the production SetFit+e5 retrain (2026-10-13).

Mirrors the lab protocol (embeddinggemma-lab experiments/email_real_prep.py):
stratified per-class 70/15/15, random.Random(7), same rounding rules — with
one change demanded by the sibling review + coordinator: BOTH copies of the
4 known dual-labeled groups are dropped before splitting (the §11-correction
protocol; user-approved for this production retrain).

Labels are path-shaped ('Jarvis/FOCUS' etc.) to match the Pi's Gmail-label
vocabulary and the deployed label_mapping.json.
"""
import ast, hashlib, json, os, random
from collections import Counter

DATA = "/workspace/projects/email_classifier_data/Jarvis"
WORK = "/workspace/projects/email_classifier/retrain_2026-10-13"
SEED = 7

def canon(r):
    mass = r.get("mass_mail", False)
    if isinstance(mass, str):
        mass = mass.strip().lower() == "true"
    att = r.get("attachment_types", [])
    if isinstance(att, str):
        try:
            att = ast.literal_eval(att)
        except (ValueError, SyntaxError):
            att = []
    return {"subject": r.get("subject", ""), "body": r.get("body", ""),
            "from": r.get("from", ""), "to": r.get("to", ""),
            "cc": r.get("cc", ""), "mass_mail": bool(mass),
            "attachment_types": list(att or [])}

def main():
    records = []          # canon + label + source provenance
    by_key = {}
    dup_groups = {}
    for fn in sorted(os.listdir(DATA)):
        if not fn.endswith(".jsonl"):
            continue
        label = f"Jarvis/{fn[:-6]}"
        for i, line in enumerate(open(os.path.join(DATA, fn)), 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r["subject"], r["body"])
            rec = canon(r) | {"label": label, "src": f"Jarvis/{fn}:{i}"}
            records.append(rec)
            by_key.setdefault(key, []).append(rec)
    # Cross-file duplicate detection (expect exactly the 4 known groups)
    for key, group in by_key.items():
        if len(group) > 1:
            dup_groups[key] = group
    # Drop BOTH copies of every dual-labeled group
    dropped_keys = {k for k, g in dup_groups.items()}
    kept = [r for r in records if (r["subject"], r["body"]) not in dropped_keys]

    by = {}
    for r in kept:
        by.setdefault(r["label"], []).append(r)
    rng = random.Random(SEED)
    train, dev, test = [], [], []
    for lab in sorted(by):
        rs = by[lab][:]
        rng.shuffle(rs)
        n = len(rs)
        n_tr = round(n * 0.70)
        n_de = max(1, round(n * 0.15)) if n >= 3 else 1
        n_te = max(1, n - n_tr - n_de)
        train += rs[:n_tr]
        dev += rs[n_tr:n_tr + n_de]
        test += rs[n_tr + n_de:]

    os.makedirs(f"{WORK}/corpus", exist_ok=True)
    for name, split in (("train", train), ("dev", dev), ("test", test)):
        with open(f"{WORK}/corpus/{name}.jsonl", "w") as f:
            for r in split:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {
        "data_repo_head": os.popen("git -C /workspace/projects/email_classifier_data rev-parse HEAD").read().strip(),
        "gross_records": len(records),
        "per_file_gross": dict(per_file_counts(records)),
        "dual_labeled_groups_dropped": [
            {"subject": k[0], "labels": sorted({g["label"] for g in g})} for k, g in sorted(dup_groups.items())
        ],
        "net_records": len(kept),
        "splits": {"train": len(train), "dev": len(dev), "test": len(test)},
        "splits_per_class": {n: dict(Counter(r["label"] for r in s)) for n, s in (("train", train), ("dev", dev), ("test", test))},
        "seed": SEED,
        "protocol": "lab 70/15/15 stratified, per-class, seed 7; both copies of dual-labeled groups dropped (conservative, lab precedent)",
    }
    with open(f"{WORK}/corpus/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))

def per_file_counts(records):
    c = Counter(r["src"].split(":")[0] + ".jsonl" for r in records)
    return dict(c)

if __name__ == "__main__":
    main()
