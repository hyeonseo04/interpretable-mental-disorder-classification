# -*- coding: utf-8 -*-
"""
core.py — 결과 로드 + 분류 파이프라인 (Table 3·Fig. 2·Fig. 3·모든 재분석의 단일 소스).

분류 규약 (논문 3.4·3.5절)
  - 다항 로지스틱 회귀, L2 (C=1.0), class_weight='balanced', lbfgs, max_iter=2000
  - StandardScaler 는 학습 폴드에서만 적합 (Pipeline)
  - 내담자 기준 GroupKFold(K=5) — 폴드 배정은 data/processed/folds.json 하나를 모든 방법이 공유
  - 학습 단위 = 세션 (세션 라벨), 평가 단위 = 내담자 (장애 우선 라벨)
  - 내담자 예측: LR 계열 = 세션 OOF 확률 평균(soft voting) → argmax
                 직접 분류(zero/few) = 세션 예측 다수결, 동률이면 가장 이른 세션의 예측
"""
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .constants import (CATEGORIES4, CLASSES, LR_C, LR_MAX_ITER, METHODS, SYMPTOM28,
                        stem as make_stem)


def normalize_sid(name):
    sid = str(name).strip().replace("_raw_", "_check_")
    return re.sub(r"(_check_)([a-z])", lambda m: m.group(1) + m.group(2).upper(), sid)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ════════════════════════════════════════════════════════════════
# 데이터셋 (매니페스트)
# ════════════════════════════════════════════════════════════════
@dataclass
class Dataset:
    sids: list
    pid: np.ndarray            # 세션별 person_id
    y: np.ndarray              # 세션 라벨
    fold: np.ndarray           # 세션별 폴드
    gold: np.ndarray           # (n_sess, 28) 골드 0~3
    persons: list              # 정렬된 내담자 목록
    person_label: dict
    sess_of: dict = field(default_factory=dict)   # person → 세션 인덱스 배열

    @property
    def person_y(self):
        return np.array([self.person_label[p] for p in self.persons])

    @property
    def person_index(self):
        return {p: i for i, p in enumerate(self.persons)}


def load_dataset(processed_dir="data/processed"):
    man = sorted(read_jsonl(os.path.join(processed_dir, "manifest.jsonl")), key=lambda r: r["session_id"])
    folds = json.load(open(os.path.join(processed_dir, "folds.json")))
    ds = Dataset(
        sids=[r["session_id"] for r in man],
        pid=np.array([r["person_id"] for r in man]),
        y=np.array([r["class"] for r in man]),
        fold=np.array([folds[r["person_id"]] for r in man]),
        gold=np.array([[r["gold"][f] for f in SYMPTOM28] for r in man], float),
        persons=sorted({r["person_id"] for r in man}),
        person_label={r["person_id"]: r["person_label"] for r in man},
    )
    ds.sess_of = {p: np.where(ds.pid == p)[0] for p in ds.persons}
    return ds


# ════════════════════════════════════════════════════════════════
# 결과 로드 (실행 기록 필드가 없는 형식도 읽을 수 있음)
# ════════════════════════════════════════════════════════════════
def result_path(results_dir, method, model, suffix):
    return os.path.join(results_dir, make_stem(method, model, suffix) + ".jsonl")


def load_scores(path, ds, keys):
    """LR 계열 결과 → (n_sess, len(keys)) 행렬, 매니페스트 순서. 누락 세션이 있으면 오류."""
    by = {}
    for r in read_jsonl(path):
        sid = normalize_sid(r["session_id"])
        by.setdefault(sid, r)                       # 중복 시 첫 기록
    miss = [s for s in ds.sids if s not in by]
    if miss:
        raise ValueError(f"{path}: 매니페스트 세션 {len(miss)}개 누락 (예: {miss[:3]})")
    X = np.array([[float(by[s]["llm"].get(k, 0)) for k in keys] for s in ds.sids])
    status = [by[s].get("parse_status", "legacy") for s in ds.sids]
    return X, status


def load_direct(path, ds):
    by = {}
    for r in read_jsonl(path):
        by.setdefault(normalize_sid(r["session_id"]), r)
    miss = [s for s in ds.sids if s not in by]
    if miss:
        raise ValueError(f"{path}: 매니페스트 세션 {len(miss)}개 누락 (예: {miss[:3]})")
    pred = np.array([by[s]["pred"] for s in ds.sids])
    ok = np.array([by[s].get("parsed_ok", True) for s in ds.sids])
    return pred, ok


# ════════════════════════════════════════════════════════════════
# 분류기
# ════════════════════════════════════════════════════════════════
def make_lr():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=LR_C, class_weight="balanced", solver="lbfgs",
                                            max_iter=LR_MAX_ITER))


def oof_proba(X, ds, feature_fn=None, rows=None):
    """GroupKFold OOF 확률 (n, 4) — 열 순서 CLASSES.
    feature_fn(Xtr, Xte) → (Xtr', Xte') : 폴드 내부에서만 적합되는 특징 변환(예: 분위수 매핑)
    rows: 부분집합 인덱스(다운샘플링 실험 등). None 이면 전체."""
    idx = np.arange(len(ds.sids)) if rows is None else np.asarray(rows)
    P = np.full((len(ds.sids), len(CLASSES)), np.nan)
    for f in sorted(set(ds.fold[idx].tolist())):
        tr, te = idx[ds.fold[idx] != f], idx[ds.fold[idx] == f]
        Xtr, Xte = X[tr], X[te]
        if feature_fn is not None:
            Xtr, Xte = feature_fn(Xtr, Xte)
        m = make_lr().fit(Xtr, ds.y[tr])
        cls = list(m.classes_)
        pr = m.predict_proba(Xte)
        P[te] = np.column_stack([pr[:, cls.index(c)] if c in cls else np.zeros(len(te))
                                 for c in CLASSES])
    return P


def person_from_proba(P, ds, rule="soft", persons=None):
    """세션 확률 → 내담자 예측.  rule: soft(평균 확률) / hard(세션 argmax 다수결) / max(최대 확률)"""
    persons = ds.persons if persons is None else persons
    out = []
    for p in persons:
        ii = ds.sess_of[p]
        if rule == "soft":
            out.append(CLASSES[int(np.argmax(P[ii].mean(0)))])
        elif rule == "max":
            out.append(CLASSES[int(np.unravel_index(np.argmax(P[ii]), P[ii].shape)[1])])
        elif rule == "hard":
            out.append(majority([CLASSES[k] for k in P[ii].argmax(1)]))
        else:
            raise ValueError(rule)
    return np.array(out)


def majority(preds):
    """다수결 — 동률이면 가장 이른 세션의 예측 (Counter.most_common 의 삽입 순서 규칙과 동일)."""
    return Counter(preds).most_common(1)[0][0]


def person_from_direct(pred, ds, persons=None):
    persons = ds.persons if persons is None else persons
    out, ties = [], 0
    for p in persons:
        seq = list(pred[ds.sess_of[p]])
        c = Counter(seq).most_common()
        if len(c) > 1 and c[0][1] == c[1][1]:
            ties += 1
        out.append(majority(seq))
    return np.array(out), ties


# ════════════════════════════════════════════════════════════════
# 한 번에: 8개 구성의 세션·내담자 예측
# ════════════════════════════════════════════════════════════════
@dataclass
class MethodPred:
    method: str
    model: str
    sess_pred: np.ndarray
    person_pred: np.ndarray
    sess_proba: np.ndarray = None   # LR 계열만
    X: np.ndarray = None            # LR 계열 점수 행렬
    parse_fail: float = 0.0
    ties: int = 0


def predict_all(ds, results_dir, suffix, models=("14b", "32b"), methods=tuple(METHODS), verbose=True):
    out = {}
    for model in models:
        for method in methods:
            path = result_path(results_dir, method, model, suffix)
            if not os.path.exists(path):
                if verbose:
                    print(f"  (없음) {path}")
                continue
            if method in ("factor28", "direct4"):
                keys = SYMPTOM28 if method == "factor28" else CATEGORIES4
                X, status = load_scores(path, ds, keys)
                P = oof_proba(X, ds)
                mp = MethodPred(method, model, np.array(CLASSES)[P.argmax(1)],
                                person_from_proba(P, ds), P, X,
                                parse_fail=float(np.mean([s not in ("ok", "legacy") for s in status])))
            else:
                pred, ok = load_direct(path, ds)
                pp, ties = person_from_direct(pred, ds)
                mp = MethodPred(method, model, pred, pp, parse_fail=float(1 - ok.mean()), ties=ties)
            out[(method, model)] = mp
    return out
