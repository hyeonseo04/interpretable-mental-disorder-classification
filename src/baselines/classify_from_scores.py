#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
classify_from_scores.py — 요인/범주 강도 벡터로부터 4범주를 분류(다항 LR).

특징 열(feature)을 입력 JSONL의 "llm" 딕셔너리에서 **자동 감지**하므로,
동일한 코드로 다음 두 방식을 공정하게 비교할 수 있다:

  Factor (제안, 28차원)      : python classify_from_scores.py session_llm_14b.jsonl
  Direct-to-LR (베이스라인, 4차원): python classify_from_scores.py session_direct4_14b.jsonl

두 실행의 유일한 차이는 입력 벡터의 granularity(28 vs 4)뿐이며,
분류기·CV·표준화·soft voting은 완전히 동일하다. 따라서 성능 차이가
"증상 수준 분해"에서 비롯됨을 통제된 조건에서 보일 수 있다(리뷰 (1) 대응).

논문 3.4/3.5절 설정 재현
  - 다항 로지스틱 회귀, L2 정규화(C=1.0), class_weight='balanced'
  - 표준화(StandardScaler)는 학습 폴드에서만 적합 → 정보 누설 방지
  - 내담자 기준 GroupKFold(K=5)  (동일인 세션이 학습/평가에 섞이지 않음)
  - 내담자 단위 예측 = 세션별 예측확률 평균(soft voting) 후 최대 확률 범주
  - 정상군 불균형 고려 → 주 지표 macro-F1, accuracy·범주별 F1 병기

출력
  - 콘솔: 세션 단위(폴드 평균±표준편차), 내담자 단위(전체 pool) 지표 + 혼동행렬
  - 파일: pred_person_{tag}.jsonl  (person_id, true, pred) → McNemar/부트스트랩 CI에 사용

사용
cd ~/Psychological_counseling

# ── classify_from_scores.py (28d Factor vs 4d Direct) ──
python src/baselines/classify_from_scores.py results/session_llm_14b.jsonl     --tag llm_14b     --out-dir results
python src/baselines/classify_from_scores.py results/session_direct4_14b.jsonl --tag direct4_14b --out-dir results
python src/baselines/classify_from_scores.py results/session_llm_32b.jsonl     --tag llm_32b     --out-dir results
python src/baselines/classify_from_scores.py results/session_direct4_32b.jsonl --tag direct4_32b --out-dir results
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

CLASSES = ["DEPRESSION", "ANXIETY", "ADDICTION", "NORMAL"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}


# ─────────────────────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────────────────────
def load_records(path):
    """JSONL을 읽어 (records, feature_names) 반환.
    feature_names는 모든 레코드의 llm 키 합집합(정렬)으로 자동 감지."""
    records = []
    feat_union = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("class") not in CLS2IDX:
                continue
            llm = r.get("llm", {})
            if not isinstance(llm, dict) or not llm:
                continue
            feat_union.update(llm.keys())
            records.append(r)
    features = sorted(feat_union)
    return records, features


def build_matrix(records, features):
    """records → (X, y, groups). 결측 요인은 0으로 채움."""
    X = np.array([[float(r["llm"].get(f, 0)) for f in features] for r in records],
                 dtype=float)
    y = np.array([CLS2IDX[r["class"]] for r in records], dtype=int)
    groups = np.array([r.get("person_id") or r.get("session_id") for r in records])
    return X, y, groups


# ─────────────────────────────────────────────────────────────
# 교차검증 분류
# ─────────────────────────────────────────────────────────────
def run_cv(X, y, groups, k=5, seed=0):
    """GroupKFold 교차검증. 세션 예측과 내담자 soft-voting 예측을 모두 수집.

    Returns
      fold_sess_f1, fold_sess_acc : 폴드별 세션 지표(list)
      person_true, person_pred    : {person: label_idx}  (각 인원은 자기 held-out 폴드에서 1회)
    """
    gkf = GroupKFold(n_splits=k)
    fold_sess_f1, fold_sess_acc = [], []
    person_true, person_pred = {}, {}

    for tr, te in gkf.split(X, y, groups):
        scaler = StandardScaler().fit(X[tr])          # 학습 폴드에서만 적합
        Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])

        clf = LogisticRegression(
            C=1.0, class_weight="balanced",
            solver="lbfgs", max_iter=2000,            # lbfgs + 다중클래스 = 다항(multinomial)
        ).fit(Xtr, y[tr])

        # 세션 단위
        pred_te = clf.predict(Xte)
        fold_sess_f1.append(f1_score(y[te], pred_te, average="macro"))
        fold_sess_acc.append(accuracy_score(y[te], pred_te))

        # 내담자 단위 soft voting: 세션 예측확률을 인원별로 평균 후 argmax
        proba = clf.predict_proba(Xte)                # (n_te, n_classes) — 열 순서 = clf.classes_
        cls_order = clf.classes_
        agg_proba = defaultdict(lambda: np.zeros(len(cls_order)))
        agg_cnt = defaultdict(int)
        for i, gi in enumerate(groups[te]):
            agg_proba[gi] += proba[i]
            agg_cnt[gi] += 1
            person_true[gi] = y[te][i]                # 동일인 라벨은 폴드 내 일관
        for gi, ps in agg_proba.items():
            mean_p = ps / agg_cnt[gi]
            person_pred[gi] = int(cls_order[int(np.argmax(mean_p))])

    return fold_sess_f1, fold_sess_acc, person_true, person_pred


# ─────────────────────────────────────────────────────────────
# 리포트
# ─────────────────────────────────────────────────────────────
def report(fold_f1, fold_acc, person_true, person_pred, tag, out_dir):
    print(f"\n===== [{tag}] 분류 성능 =====")
    print(f"[세션] macro-F1 {np.mean(fold_f1):.3f} ± {np.std(fold_f1):.3f}"
          f" · acc {np.mean(fold_acc):.3f} ± {np.std(fold_acc):.3f}  (GroupKFold {len(fold_f1)}폴드)")

    pids = sorted(person_true)
    yt = np.array([person_true[p] for p in pids])
    yp = np.array([person_pred[p] for p in pids])

    macro = f1_score(yt, yp, average="macro")
    acc = accuracy_score(yt, yp)
    per_class = f1_score(yt, yp, average=None, labels=list(range(len(CLASSES))))
    print(f"[사람] macro-F1 {macro:.3f} · acc {acc:.3f}  (n={len(pids)}명, soft voting)")
    print("       범주별 F1: " + "  ".join(
        f"{CLASSES[i][:3]}={per_class[i]:.3f}" for i in range(len(CLASSES))))

    cm = confusion_matrix(yt, yp, labels=list(range(len(CLASSES))))
    print("\n[사람 혼동행렬] (행=정답, 열=예측)")
    print(" " * 12 + "".join(c[:4].rjust(8) for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(c[:11].ljust(12) + "".join(str(cm[i, j]).rjust(8) for j in range(len(CLASSES))))

    # 내담자 예측 저장 → McNemar / 부트스트랩 CI 재료
    pred_path = os.path.join(out_dir, f"pred_person_{tag}.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for p in pids:
            f.write(json.dumps({
                "person_id": p,
                "true": CLASSES[person_true[p]],
                "pred": CLASSES[person_pred[p]],
            }, ensure_ascii=False) + "\n")
    print(f"\n[저장] 내담자 예측 → {pred_path}")
    print("       (두 방식의 pred_person_*.jsonl로 McNemar·부트스트랩 CI를 산출)")


def main():
    ap = argparse.ArgumentParser(description="Score-vector classifier (feature-agnostic).")
    ap.add_argument("scores_jsonl", help="extract_*.py 출력 JSONL (llm 딕셔너리 포함)")
    ap.add_argument("--tag", default=None, help="출력 파일 태그(기본: 입력 파일명)")
    ap.add_argument("--k", type=int, default=5, help="GroupKFold 폴드 수")
    ap.add_argument("--out-dir", default=".", help="예측 파일 저장 디렉토리")
    args = ap.parse_args()

    tag = args.tag or os.path.splitext(os.path.basename(args.scores_jsonl))[0]

    records, features = load_records(args.scores_jsonl)
    X, y, groups = build_matrix(records, features)
    n_person = len(set(groups))
    print(f"[로드] 세션 {len(records)} · 인원 {n_person} · 특징 {len(features)}차원")
    print(f"       features = {features}")

    fold_f1, fold_acc, ptrue, ppred = run_cv(X, y, groups, k=args.k)
    report(fold_f1, fold_acc, ptrue, ppred, tag, args.out_dir)


if __name__ == "__main__":
    main()