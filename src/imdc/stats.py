# -*- coding: utf-8 -*-
"""
stats.py — 지표·검정·부트스트랩 (분석 전체의 단일 구현).

부트스트랩 단위 규약 (심사위원 C-(1), A-1 대응)
  - 모든 CI 는 **내담자(cluster) 단위 리샘플링**이다.
      · 내담자 단위 지표: 내담자 207명을 복원추출 (내담자당 관측 1개 → 클러스터 = 관측)
      · 세션 단위 지표 : 내담자를 복원추출한 뒤 그 내담자의 세션을 전부 포함
    (세션을 독립적으로 리샘플하면 같은 내담자 세션 간 상관을 무시해 CI 가 과소추정된다.)
  - 두 방법/모델 비교는 같은 리샘플 인덱스를 공유하는 **paired** 부트스트랩.
  - 예측은 고정하고 평가 표본만 리샘플한다 (재학습 없음).
"""
import numpy as np
from scipy.stats import binomtest, chi2
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from .constants import CLASSES


def macro_f1(yt, yp, labels=CLASSES):
    return f1_score(yt, yp, average="macro", labels=labels, zero_division=0)


def per_class(yt, yp, labels=CLASSES):
    P, R, F, S = precision_recall_fscore_support(yt, yp, labels=labels, zero_division=0)
    return {c: {"P": P[i], "R": R[i], "F1": F[i], "n": int(S[i])} for i, c in enumerate(labels)}


def summary(yt, yp, labels=CLASSES):
    yt, yp = np.asarray(yt), np.asarray(yp)
    d = {"macro_f1": macro_f1(yt, yp, labels), "acc": accuracy_score(yt, yp)}
    pc = per_class(yt, yp, labels)
    for c in labels:
        d[f"F1_{c}"] = pc[c]["F1"]
    # 정상군·선별 지표 (A-2)
    if "NORMAL" in labels:
        d["NOR_precision"] = pc["NORMAL"]["P"]
        d["NOR_recall"] = pc["NORMAL"]["R"]
        dis_t, dis_p = yt != "NORMAL", yp != "NORMAL"
        d["disorder_sensitivity"] = (dis_t & dis_p).sum() / max(dis_t.sum(), 1)
        d["disorder_to_normal"] = (dis_t & ~dis_p).sum() / max(dis_t.sum(), 1)
        mask = yt != "NORMAL"
        d["macro_f1_3cls_posthoc"] = f1_score(yt[mask], yp[mask], average="macro",
                                              labels=[c for c in labels if c != "NORMAL"],
                                              zero_division=0)
    return d


# ── 부트스트랩 ──────────────────────────────────────────────────
def cluster_indices(groups):
    """groups(관측별 클러스터 id) → (unique ids, {id: 관측 인덱스 배열})"""
    groups = np.asarray(groups)
    uniq = np.array(sorted(set(groups.tolist())))
    idx = {g: np.where(groups == g)[0] for g in uniq}
    return uniq, idx


def cluster_boot_samples(groups, n_boot, rng):
    uniq, idx = cluster_indices(groups)
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        yield np.concatenate([idx[uniq[k]] for k in pick])


def boot_ci(stat_fn, groups, n_boot, rng, alpha=0.05):
    """stat_fn(index_array) → float. 내담자 클러스터 부트스트랩 percentile CI."""
    vals = np.array([stat_fn(ii) for ii in cluster_boot_samples(groups, n_boot, rng)])
    lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), vals


def paired_boot_diff(stat_a, stat_b, groups, n_boot, rng):
    """Δ = stat_a − stat_b 의 paired 클러스터 부트스트랩. returns (lo, hi, p_two_sided)"""
    d = []
    for ii in cluster_boot_samples(groups, n_boot, rng):
        d.append(stat_a(ii) - stat_b(ii))
    d = np.array(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(lo), float(hi), float(min(p, 1.0))


# ── 검정 ────────────────────────────────────────────────────────
def mcnemar(a_correct, b_correct):
    """b+c<25 → exact binomial, 아니면 연속성 보정 χ²."""
    a = np.asarray(a_correct, bool); b_ = np.asarray(b_correct, bool)
    b = int(np.sum(a & ~b_)); c = int(np.sum(~a & b_))
    if b + c == 0:
        return {"a_only": b, "b_only": c, "p": 1.0, "test": "none"}
    if b + c < 25:
        return {"a_only": b, "b_only": c, "p": binomtest(min(b, c), b + c, 0.5).pvalue, "test": "exact"}
    return {"a_only": b, "b_only": c, "p": float(chi2.sf((abs(b - c) - 1) ** 2 / (b + c), 1)),
            "test": "chi2_cc"}


def holm(pvals: dict):
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, prev, out = len(items), 0.0, {}
    for rank, (k, p) in enumerate(items):
        adj = max(min(p * (m - rank), 1.0), prev)
        out[k] = adj; prev = adj
    return out


def bh(pvals):
    p = np.asarray(pvals, float)
    m = len(p)
    if m == 0:
        return p
    order = p.argsort()
    r = p[order] * m / np.arange(1, m + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    out = np.empty(m); out[order] = np.clip(r, 0, 1)
    return out
