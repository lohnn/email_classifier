#!/usr/bin/env python3
"""Held-out dev/test evaluation for the production retrain — lab-protocol
numbers (accuracy + per-class + ms/email) plus the production UNSURE gate
mayı (classify.is_unsure_classification thresholds) for the scoreboard."""
import json, os, sys, time
import numpy as np

WORK = "/workspace/projects/email_classifier/retrain_2026-10-13"
SP = "/workspace/projects/email_classifier/email_classifier_brain"
sys.path.insert(0, SP)
import classify as brain_classify      # noqa: E402  (loads the NEW model below)
from setfit import SetFitModel         # noqa: E402

MODEL_DIR = os.environ.get("EVAL_MODEL_DIR", f"{WORK}/model")

def to_rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]

def main():
    model = SetFitModel.from_pretrained(MODEL_DIR)
    labels = sorted({v for v in json.load(open(f"{MODEL_DIR}/label_mapping.json")).values()})
    lm = {v: int(k) for k, v in json.load(open(f"{MODEL_DIR}/label_mapping.json")).items()}
    import config as brain_config
    report = {"model_dir": MODEL_DIR, "setfit_created_by": "1.2.0/ST5.7.0/sklearn1.9.1/torch2.14.1 (this box)",
              "pi_runtime": "setfit 1.1.0/ST 3.4.1/sklearn 1.8.0/torch 2.3.0 (requirements_pi.txt)",
              "thresholds": {"confidence": brain_config.UNSURE_CONFIDENCE_THRESHOLD,
                             "delta": brain_config.UNSURE_DELTA_THRESHOLD}, "splits": {}}
    for split in ("dev", "test"):
        rows = to_rows(f"{WORK}/corpus/{split}.jsonl")
        texts = [brain_config.format_model_input(
            subject=r["subject"], body=r["body"], sender=r["from"], to=r["to"],
            cc=r["cc"], mass_mail=r["mass_mail"], attachment_types=r["attachment_types"])
            for r in rows]
        t0 = time.perf_counter()
        probs = np.asarray(model.predict_proba(texts))
        ms = (time.perf_counter() - t0) * 1000
        idx = probs.argmax(1)
        preds = [labels[i] for i in idx]
        n = len(rows)
        acc = sum(p == r["label"] for p, r in zip(preds, rows)) / n
        unsure = sum(1 for row in probs if brain_classify.is_unsure_classification(row)) / n
        per_class = {}
        for lab in labels:
            trues = [i for i, r in enumerate(rows) if r["label"] == lab]
            if not trues:
                continue
            tp = sum(preds[i] == lab for i in trues)
            per_class[lab] = {"n": len(trues), "correct": tp}
        conf = [[{"label": labels[j], "p": round(float(probs[i][j]), 4)}
                 for j in probs[i].argsort()[::-1][:2]] for i in range(n)]
        report["splits"][split] = {
            "accuracy": round(acc, 4), "n": n,
            "unsure_rate": round(unsure, 4), "ms_per_email": round(ms / n, 1),
            "per_class": per_class,
            "rows": [{"subject": r["subject"][:70], "true": r["label"], "pred": preds[i],
                      "top2": conf[i], "correct": preds[i] == r["label"]}
                     for i, r in enumerate(rows)],
        }
        print(f"{split}: {acc:.0%} unsure={unsure:.0%} ({n} rows, {ms/n:.0f}ms/email)")
        for lab, s in per_class.items():
            print(f"   {lab}: {s['correct']}/{s['n']}")
    out = f"{WORK}/eval/report.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(report, open(out, "w"), indent=2, ensure_ascii=False)
    print("report:", out)

if __name__ == "__main__":
    main()
