#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_tables.py — 논문 Table 3(분류 성능) / Table 4(추출 충실성)를 CSV로 저장.

  Table 3 : Factor(로지스틱 회귀, soft voting) + zero/few 직접분류의
            내담자 단위 macro-F1 / accuracy / per-class F1 (14B·32B)
  Table 4 : 요인별 추출 충실성 (Spearman ρ, QWK, recall, gold≥2)  — 14B·32B

  데이터 규약: session_id 정규화, person_id 대문자, 내담자 라벨=장애 우선.

사용:
  uv run --with scikit-learn --with scipy make_tables.py \
      --result-dir result --out-dir result --min-pos 30
"""

import argparse
import csv
import json
import os
import re
from collections import defaultdict, Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

CLASSES = ["ADDICTION", "ANXIETY", "DEPRESSION", "NORMAL"]
SYMPTOM28 = [
    "depressive_mood", "worthlessness", "guilt", "impaired_cognition", "suicidal", "anhedonia",
    "psychomotor_changes", "weight_appetite", "sleep_disturbance", "fatigue",
    "anxiety_mood", "derealization", "perceived_loss_of_control", "anxiety_control",
    "concentration", "avoidance", "physical_symptoms", "irritability",
    "loss_of_control", "craving", "lying", "tolerance", "withdrawal", "salience",
    "resource_investment", "daily_functioning", "social_problems", "negative_consequences",
]


def norm(s):
    s = str(s).strip().replace("_raw_", "_check_")
    return re.sub(r"(_check_)([a-z])", lambda m: m.group(1) + m.group(2).upper(), s)


def read_jsonl(path):
    rows, seen = [], set()
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        sid = norm(r["session_id"])
        if sid in seen:
            continue
        seen.add(sid)
        r["session_id"] = sid
        r["person_id"] = str(r["person_id"]).upper()
        rows.append(r)
    return rows


def person_true_map(rows, cls_key):
    """내담자 라벨 = 장애 우선(장애 세션 하나라도 있으면 그 장애, 아니면 NORMAL)."""
    pc = defaultdict(set)
    for r in rows:
        pc[r["person_id"]].add(r[cls_key])
    out = {}
    for p, cs in pc.items():
        dis = cs - {"NORMAL"}
        out[p] = sorted(dis)[0] if dis else "NORMAL"
    return out


def factor_person_pred(path, folds=5, seed=0):
    """Factor: GroupKFold OOF proba → 사람 soft voting 예측/정답."""
    rows = read_jsonl(path)
    X = np.array([[r["llm"][f] for f in SYMPTOM28] for r in rows], float)
    y = np.array([r["class"] for r in rows])
    grp = np.array([r["person_id"] for r in rows])
    labels = sorted(set(y))
    oof = np.zeros((len(rows), len(labels)))
    for tr, te in GroupKFold(folds).split(X, y, grp):
        p = make_pipeline(StandardScaler(),
                          LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")).fit(X[tr], y[tr])
        cls = list(p.classes_)
        oof[te] = p.predict_proba(X[te])[:, [cls.index(c) for c in labels]]
    ptrue = person_true_map(rows, "class")
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[r["person_id"]].append(oof[i])
    pids = list(by)
    yt = [ptrue[p] for p in pids]
    yp = [labels[np.array(by[p]).mean(0).argmax()] for p in pids]
    return yt, yp


def direct_person_pred(path):
    """direct: 사람 다수결 예측/정답."""
    rows = read_jsonl(path)
    by = defaultdict(list)
    for r in rows:
        by[r["person_id"]].append(r["pred"])
    ptrue = person_true_map(rows, "true")
    pids = list(by)
    yt = [ptrue[p] for p in pids]
    yp = [Counter(by[p]).most_common(1)[0][0] for p in pids]
    return yt, yp


def metrics_row(yt, yp):
    macro = f1_score(yt, yp, average="macro", labels=CLASSES, zero_division=0)
    acc = accuracy_score(yt, yp)
    _, _, F, _ = precision_recall_fscore_support(yt, yp, labels=CLASSES, zero_division=0)
    return macro, acc, F


def build_table3(rdir):
    header = ["Model", "Method", "macro_F1", "Acc"] + [c[:3] for c in CLASSES]
    table = [header]
    for m in ["14b", "32b"]:
        for method, fn in [
            ("Zero-shot", f"session_direct_zeroshot_{m}.jsonl"),
            ("Few-shot",  f"session_direct_fewshot_{m}.jsonl"),
            ("Factor",    f"session_llm_{m}.jsonl"),
        ]:
            path = os.path.join(rdir, fn)
            if not os.path.exists(path):
                print(f"  (없음) {path}"); continue
            if method == "Factor":
                yt, yp = factor_person_pred(path)
            else:
                yt, yp = direct_person_pred(path)
            macro, acc, F = metrics_row(yt, yp)
            table.append([m.upper(), method, f"{macro:.3f}", f"{acc:.3f}",
                          *[f"{x:.3f}" for x in F]])
    return table


def build_table4(rdir, min_pos):
    gold = {norm(json.loads(l)["session_id"]): json.loads(l)["gold"]
            for l in open(os.path.join(rdir, "session_gold.jsonl")) if l.strip()}
    llm = {}
    for m in ["14b", "32b"]:
        p = os.path.join(rdir, f"session_llm_{m}.jsonl")
        llm[m] = {norm(json.loads(l)["session_id"]): json.loads(l)["llm"]
                  for l in open(p) if l.strip()} if os.path.exists(p) else {}

    header = ["factor", "gold_pos", "rho_14b", "rho_32b",
              "qwk_14b", "qwk_32b", "recall_14b", "recall_32b", "verified"]
    table = [header]
    for f in SYMPTOM28:
        common = sorted(set(gold) & set(llm.get("14b", {})))
        g = [gold[s].get(f, 0) for s in common]
        pos2 = sum(1 for x in g if x >= 2)
        row = [f, pos2]
        stats = {}
        for m in ["14b", "32b"]:
            cm = sorted(set(gold) & set(llm.get(m, {})))
            gg = [gold[s].get(f, 0) for s in cm]
            ll = [llm[m][s].get(f, 0) for s in cm]
            rho = (spearmanr(gg, ll).correlation
                   if len(set(gg)) > 1 and len(set(ll)) > 1 else float("nan"))
            try:
                qwk = cohen_kappa_score(gg, ll, weights="quadratic", labels=[0, 1, 2, 3])
            except Exception:
                qwk = float("nan")
            pair = [(a, b) for a, b in zip(gg, ll) if a >= 2]
            rec = (sum(1 for a, b in pair if b >= 2) / len(pair)) if pair else float("nan")
            stats[m] = (rho, qwk, rec)
        row += [f"{stats['14b'][0]:.3f}", f"{stats['32b'][0]:.3f}",
                f"{stats['14b'][1]:.3f}", f"{stats['32b'][1]:.3f}",
                f"{stats['14b'][2]:.2f}", f"{stats['32b'][2]:.2f}",
                "Y" if pos2 >= min_pos else "N"]
        table.append(row)
    # ρ(14b) 내림차순 정렬(검증 대상 먼저), 제외는 뒤로
    body = table[1:]
    body.sort(key=lambda r: (r[-1] != "Y", -float(r[2]) if r[2] != "nan" else 0))
    return [header] + body


def save_csv(table, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(table)
    print(f"저장: {path} ({len(table)-1} 행)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", default="result")
    ap.add_argument("--out-dir", default="result")
    ap.add_argument("--min-pos", type=int, default=30)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("[Table 3] 분류 성능")
    t3 = build_table3(args.result_dir)
    for row in t3:
        print("  " + " | ".join(str(x).rjust(9) for x in row))
    save_csv(t3, os.path.join(args.out_dir, "table3_performance.csv"))

    print("\n[Table 4] 추출 충실성")
    t4 = build_table4(args.result_dir, args.min_pos)
    save_csv(t4, os.path.join(args.out_dir, "table4_faithfulness.csv"))
    dropped = [r[0] for r in t4[1:] if r[-1] == "N"]
    print(f"  검증 제외({len(dropped)}): {dropped}")


if __name__ == "__main__":
    main()