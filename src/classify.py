#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
classify.py — LLM 28-factor → 다항 로지스틱 회귀 4-class 분류

방법(논문 3.4~3.5):
  · 피처 : LLM 28-factor 0~3 순서형 값 그대로
  · 모델 : 다항 로지스틱 회귀 (L2, class_weight='balanced'), StandardScaler(폴드 내 적합)
  · 분할 : person_id 기준 GroupKFold(K=5)  → 같은 사람 세션이 train/test 에 안 섞임
  · 학습 : 세션 단위 (세션 1개 = 샘플 1개)
  · 평가 : 세션 단위 + 사람 단위(soft voting), macro-F1 / accuracy

데이터 규약 (vllm_extract.py / gold_aggregate.py 와 동일):
  · session_id 정규화(_raw_→_check_, person 대문자), person_id 대문자
  · 손상/누락 세션은 이미 추출 단계에서 제외되어 있음(입력 jsonl 이 정본)

사용:
  uv run --with scikit-learn classify.py \
      --input ../results/session_llm_32b.jsonl --folds 5
"""

import argparse
import json
import re
from collections import defaultdict, Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

SYMPTOM28 = [
    "depressive_mood", "worthlessness", "guilt", "impaired_cognition", "suicidal", "anhedonia",
    "psychomotor_changes", "weight_appetite", "sleep_disturbance", "fatigue",
    "anxiety_mood", "derealization", "perceived_loss_of_control", "anxiety_control",
    "concentration", "avoidance", "physical_symptoms", "irritability",
    "loss_of_control", "craving", "lying", "tolerance", "withdrawal", "salience",
    "resource_investment", "daily_functioning", "social_problems", "negative_consequences",
]


def normalize_sid(name: str) -> str:
    sid = name.strip().replace("_raw_", "_check_")
    return re.sub(r"(_check_)([a-z])", lambda m: m.group(1) + m.group(2).upper(), sid)


def load_rows(path: str):
    """입력 jsonl 로드 + 정규화 + 중복 제거(정본 보장)."""
    rows, seen = [], set()
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        sid = normalize_sid(r["session_id"])
        if sid in seen:
            continue
        seen.add(sid)
        r["session_id"] = sid
        r["person_id"] = str(r["person_id"]).upper()
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="result/session_llm_14b.jsonl")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = load_rows(args.input)
    X = np.array([[r["llm"][f] for f in SYMPTOM28] for r in rows], dtype=float)
    y = np.array([r["class"] for r in rows])
    groups = np.array([r["person_id"] for r in rows])
    labels = sorted(set(y))
    print(f"세션 {len(rows)} · 사람 {len(set(groups))} · 클래스 {labels}")

    # 한 사람 = 한 범주인지 검증 (다르면 person_true 로직 재검토 필요)
    pc = defaultdict(set)
    for r in rows:
        pc[r["person_id"]].add(r["class"])
    multi = {p: cs for p, cs in pc.items() if len(cs) > 1}
    if multi:
        print(f"⚠ 여러 범주를 가진 사람 {len(multi)}명: {dict(list(multi.items())[:5])} ...")
    else:
        print("✓ 모든 사람이 단일 범주")

    # ── out-of-fold 예측 (사람 누수 없음) ──
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"),
    )
    proba = cross_val_predict(
        clf, X, y, cv=GroupKFold(args.folds), groups=groups, method="predict_proba"
    )

    # ── 세션 단위 ──
    pred_sess = np.array([labels[i] for i in proba.argmax(1)])
    print(f"\n[세션] macro-F1 {f1_score(y, pred_sess, average='macro'):.3f}"
          f" · acc {accuracy_score(y, pred_sess):.3f}")

    # ── 사람 단위 (soft voting) ──
    # 사람 정답: 장애 세션이 하나라도 있으면 그 장애, 전부 NORMAL이면 NORMAL
    _pc = defaultdict(set)
    for r in rows:
        _pc[r["person_id"]].add(r["class"])
    person_true = {p: (cs - {"NORMAL"}).pop() if (cs - {"NORMAL"}) else "NORMAL"
                for p, cs in _pc.items()}
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[r["person_id"]].append(proba[i])
    pids = list(by)
    tt = [person_true[p] for p in pids]
    pp = [labels[np.array(by[p]).mean(0).argmax()] for p in pids]   # soft voting
    print(f"[사람] macro-F1 {f1_score(tt, pp, average='macro'):.3f}"
          f" · acc {accuracy_score(tt, pp):.3f} (n={len(pids)}명, soft voting)")

    # ── 사람 혼동행렬 ──
    cm = confusion_matrix(tt, pp, labels=labels)
    print("\n[사람 혼동행렬] (행=정답, 열=예측)")
    print(" " * 12 + "".join(c[:4].rjust(7) for c in labels))
    for i, c in enumerate(labels):
        print(c[:11].ljust(12) + "".join(str(cm[i, j]).rjust(7) for j in range(len(labels))))

    # ── 클래스별 상위 기여 factor (전체 적합, 해석 참고용) ──
    #    ※ 부트스트랩 신뢰구간 기반 유의성은 stats.py 에서 산출.
    clf.fit(X, y)
    coef = clf.named_steps["logisticregression"].coef_
    cl_ = clf.named_steps["logisticregression"].classes_
    print("\n[클래스별 상위 기여 factor] (표준화 계수 상위 5, 참고용)")
    for ci, c in enumerate(cl_):
        top = np.argsort(-coef[ci])[:5]
        print(f"  {c:11}: " + ", ".join(f"{SYMPTOM28[t]}({coef[ci][t]:+.2f})" for t in top))


if __name__ == "__main__":
    main()
    
    