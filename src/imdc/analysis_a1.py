# -*- coding: utf-8 -*-
"""
analysis_a1.py — 심사위원 A-1: 14B vs 32B 성능 차이의 원인 (A1-1 ~ A1-5).

  python -m imdc.analysis_a1 --results-dir results --suffix _full --out-dir analysis_out/a1

  A1-1 요인별 precision / recall / F1 (예측≥1 vs 골드≥1)
  A1-2 점수 분포: 0~3 비율, 평균, SD, 엔트로피, 사용 범주 수 + Wilcoxon(요인 쌍, n=28)
  A1-3 양성/음성 (세션,요인) 쌍: 양성 평균점수, miss rate, FP rate, AUC(=판별력), Cliff's δ
  A1-4 통계 검증: McNemar(내담자), ΔmacroF1 paired 클러스터 부트스트랩 CI, 폴드별 Δ + Wilcoxon
  A1-5 분위수 매핑: 32B 요인 점수를 학습 폴드의 14B 분포로 매핑 후 LR 재학습

해석 규칙 (계획서 A1-0)
  표준화(z-score)가 학습 폴드에서 적용되므로 점수의 '전반적 하향 이동'만으로는 LR 성능이 변하지 않는다.
  → 32B 가 miss↑·FP↓ 이면서 AUC 가 비슷하면 '임계값 이동'(성능 하락의 원인 아님),
     AUC 까지 낮으면 '판별 신호 손실'. A1-5 와 교차 확인.
  ※ A1-5 주의: 4단계 이산 점수에서 분위수 매핑은 단조 변환이라, 표준화 후에는 '수준 간 간격'만 바뀐다.
     동점(대부분 0)이 많으면 분산을 복원할 수 없으므로 '회복 안 됨'이 곧 판별력 저하의 증거는 아니다 — A1-3 과 함께 해석.
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import rankdata, wilcoxon

from . import stats as S
from .constants import CLASSES, N_BOOT, SEED, SYMPTOM28
from .core import load_dataset, oof_proba, person_from_proba, predict_all
from .paper import save_csv

MODELS = ("14b", "32b")


def entropy(counts):
    p = np.asarray(counts, float); p = p[p > 0] / p.sum()
    return float(-(p * np.log2(p)).sum())


def fast_auc(score, pos):
    """Mann–Whitney AUC (동점 0.5 처리). pos: bool"""
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(score)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def a1_1(ds, X, out):
    rows = [["factor", "n_gold_pos"] + [f"{k}_{m}" for m in MODELS for k in ("P", "R", "F1")]]
    agg = {m: [] for m in MODELS}
    for fi, f in enumerate(SYMPTOM28):
        g = ds.gold[:, fi] >= 1
        row = [f, int(g.sum())]
        for m in MODELS:
            p = X[m][:, fi] >= 1
            tp = (p & g).sum()
            P = tp / p.sum() if p.sum() else np.nan
            R = tp / g.sum() if g.sum() else np.nan
            F = 2 * P * R / (P + R) if P == P and R == R and (P + R) > 0 else np.nan
            row += [f"{P:.3f}", f"{R:.3f}", f"{F:.3f}"]
            agg[m].append((P, R, F))
        rows.append(row)
    save_csv(rows, os.path.join(out, "a1_1_factor_prf.csv"))
    res = {m: dict(zip(("P", "R", "F1"), np.nanmean(np.array(agg[m], float), 0).round(3).tolist()))
           for m in MODELS}
    print(f"  A1-1 요인 평균  14B {res['14b']}  |  32B {res['32b']}")
    dP, dR = res["32b"]["P"] - res["14b"]["P"], res["32b"]["R"] - res["14b"]["R"]
    res["pattern"] = ("보수성(precision↑ recall↓)" if dP > 0 > dR else
                      "품질 저하(precision↓ recall↓)" if dP < 0 and dR < 0 else "혼합")
    print(f"       → 32B−14B: ΔP {dP:+.3f}, ΔR {dR:+.3f} → {res['pattern']}")
    return res


def a1_2(ds, X, out):
    rows = [["factor"] + [f"{k}_{m}" for m in MODELS
                          for k in ("p0", "p1", "p2", "p3", "mean", "sd", "entropy", "n_levels")]]
    per = {m: {"mean": [], "sd": [], "ent": []} for m in MODELS}
    for fi, f in enumerate(SYMPTOM28):
        row = [f]
        for m in MODELS:
            v = X[m][:, fi]
            cnt = [(v == k).sum() for k in range(4)]
            e = entropy(cnt)
            row += [f"{c / len(v):.3f}" for c in cnt] + [f"{v.mean():.3f}", f"{v.std():.3f}", f"{e:.3f}",
                                                          int(sum(c > 0 for c in cnt))]
            per[m]["mean"].append(v.mean()); per[m]["sd"].append(v.std()); per[m]["ent"].append(e)
        rows.append(row)
    save_csv(rows, os.path.join(out, "a1_2_score_distribution.csv"))
    overall = {m: {f"p{k}": float((X[m] == k).mean()) for k in range(4)} for m in MODELS}
    res = {"overall": overall}
    for k in ("mean", "sd", "ent"):
        a, b = np.array(per["14b"][k]), np.array(per["32b"][k])
        p = wilcoxon(a, b).pvalue if np.any(a != b) else 1.0
        res[f"wilcoxon_{k}"] = {"median_14b": float(np.median(a)), "median_32b": float(np.median(b)),
                                "p": float(p)}
    print(f"  A1-2 점수 0 비율  14B {overall['14b']['p0']:.3f} · 32B {overall['32b']['p0']:.3f} "
          f"(Δ {100 * (overall['32b']['p0'] - overall['14b']['p0']):+.1f}%p)")
    print(f"       요인별 SD 중앙값 14B {res['wilcoxon_sd']['median_14b']:.3f} · 32B "
          f"{res['wilcoxon_sd']['median_32b']:.3f} (Wilcoxon p={res['wilcoxon_sd']['p']:.3g})"
          "  ← 표준화로 복구되지 않는 '분산 축소' 지표")
    return res


def a1_3(ds, X, out, n_boot, rng):
    rows = [["factor", "n_pos"] + [f"{k}_{m}" for m in MODELS
                                   for k in ("mean_pos", "miss_rate", "fp_rate", "auc", "cliffs_delta")]]
    aucs = {m: [] for m in MODELS}
    pooled = {}
    for fi, f in enumerate(SYMPTOM28):
        g = ds.gold[:, fi] >= 1
        row = [f, int(g.sum())]
        for m in MODELS:
            v = X[m][:, fi]
            auc = fast_auc(v, g)
            aucs[m].append(auc)
            row += [f"{v[g].mean():.3f}" if g.any() else "nan",
                    f"{(v[g] == 0).mean():.3f}" if g.any() else "nan",
                    f"{(v[~g] >= 1).mean():.3f}", f"{auc:.3f}", f"{2 * auc - 1:.3f}"]
        rows.append(row)
    save_csv(rows, os.path.join(out, "a1_3_pos_neg_tendency.csv"))
    G = ds.gold >= 1
    for m in MODELS:
        pooled[m] = {"mean_pos": float(X[m][G].mean()), "miss_rate": float((X[m][G] == 0).mean()),
                     "fp_rate": float((X[m][~G] >= 1).mean()),
                     "mean_auc": float(np.nanmean(aucs[m]))}
    # 평균 AUC 차이의 paired 클러스터 부트스트랩
    valid = [fi for fi in range(28) if 0 < G[:, fi].sum() < len(G)]

    def mean_auc(m):
        return lambda ii: np.nanmean([fast_auc(X[m][ii, fi], G[ii, fi]) for fi in valid])

    lo, hi, p = S.paired_boot_diff(mean_auc("14b"), mean_auc("32b"), ds.pid, n_boot, rng)
    a, b = np.array(aucs["14b"])[valid], np.array(aucs["32b"])[valid]
    res = {"pooled": pooled, "delta_mean_auc_14b_minus_32b": {
        "est": pooled["14b"]["mean_auc"] - pooled["32b"]["mean_auc"], "ci": [lo, hi], "p_boot": p},
        "wilcoxon_factor_auc_p": float(wilcoxon(a, b).pvalue) if np.any(a != b) else 1.0}
    for m in MODELS:
        d = pooled[m]
        print(f"  A1-3 [{m}] 양성 평균 {d['mean_pos']:.2f} · miss {d['miss_rate']:.3f} · "
              f"FP {d['fp_rate']:.3f} · 평균 AUC {d['mean_auc']:.3f}")
    print(f"       ΔAUC(14B−32B) {res['delta_mean_auc_14b_minus_32b']['est']:+.3f} "
          f"[{lo:+.3f}, {hi:+.3f}] (클러스터 부트스트랩)")
    return res


def a1_4(ds, preds, out, n_boot, rng):
    yt = ds.person_y
    a, b = preds[("factor28", "14b")], preds[("factor28", "32b")]
    mc = S.mcnemar(a.person_pred == yt, b.person_pred == yt)
    lo, hi, p = S.paired_boot_diff(lambda ii: S.macro_f1(yt[ii], a.person_pred[ii]),
                                   lambda ii: S.macro_f1(yt[ii], b.person_pred[ii]),
                                   ds.persons, n_boot, rng)
    # 폴드별 (내담자 단위, 각 폴드의 평가 내담자)
    pfold = np.array([ds.fold[ds.sess_of[p][0]] for p in ds.persons])
    fa = [S.macro_f1(yt[pfold == k], a.person_pred[pfold == k]) for k in sorted(set(pfold))]
    fb = [S.macro_f1(yt[pfold == k], b.person_pred[pfold == k]) for k in sorted(set(pfold))]
    d = np.array(fa) - np.array(fb)
    wp = float(wilcoxon(fa, fb).pvalue) if np.any(d != 0) else 1.0
    est = S.macro_f1(yt, a.person_pred) - S.macro_f1(yt, b.person_pred)
    res = {"delta_macro_f1": est, "ci": [lo, hi], "p_boot": p, "mcnemar": mc,
           "fold_f1_14b": fa, "fold_f1_32b": fb, "fold_wilcoxon_p": wp}
    print(f"  A1-4 ΔmacroF1(14B−32B) {est:+.3f} [{lo:+.3f}, {hi:+.3f}] · McNemar p={mc['p']:.3g} "
          f"(14B만 정답 {mc['a_only']}, 32B만 정답 {mc['b_only']}) · 폴드 Wilcoxon p={wp:.3g} (n=5, 참고)")
    if lo <= 0 <= hi:
        print("       → 유의한 차이 없음: 본문 '보수적 추출' 설명은 '가설'로 완화해야 함 (검정력 제약 명시)")
    return res


def quantile_map_fn(X14_all):
    """학습 폴드의 32B 값 CDF(mid-rank) → 같은 학습 폴드 14B 분포의 분위수로 매핑."""
    def fn(idx_tr, idx_te, X32):
        Xtr, Xte = X32[idx_tr].copy(), X32[idx_te].copy()
        for j in range(X32.shape[1]):
            src, tgt = X32[idx_tr, j], X14_all[idx_tr, j]
            n = len(src)
            def u(v):
                return ((src < v[:, None]).sum(1) + 0.5 * (src == v[:, None]).sum(1)) / n
            Xtr[:, j] = np.quantile(tgt, u(X32[idx_tr, j]))
            Xte[:, j] = np.quantile(tgt, np.clip(u(X32[idx_te, j]), 0, 1))
        return Xtr, Xte
    return fn


def a1_5(ds, preds, out):
    X14, X32 = preds[("factor28", "14b")].X, preds[("factor28", "32b")].X
    qm = quantile_map_fn(X14)
    yt = ds.person_y
    # oof_proba 의 feature_fn 은 (Xtr, Xte) 만 받으므로 인덱스를 클로저로 전달
    P = np.full((len(ds.sids), len(CLASSES)), np.nan)
    from .core import make_lr
    for f in sorted(set(ds.fold.tolist())):
        tr, te = np.where(ds.fold != f)[0], np.where(ds.fold == f)[0]
        Xtr, Xte = qm(tr, te, X32)
        m = make_lr().fit(Xtr, ds.y[tr]); cls = list(m.classes_)
        P[te] = m.predict_proba(Xte)[:, [cls.index(c) for c in CLASSES]]
    mapped = person_from_proba(P, ds)
    res = {"macro_f1_14b": S.macro_f1(yt, preds[("factor28", "14b")].person_pred),
           "macro_f1_32b": S.macro_f1(yt, preds[("factor28", "32b")].person_pred),
           "macro_f1_32b_quantile_mapped": S.macro_f1(yt, mapped)}
    gap = res["macro_f1_14b"] - res["macro_f1_32b"]
    rec = res["macro_f1_32b_quantile_mapped"] - res["macro_f1_32b"]
    res["recovered_fraction_of_gap"] = rec / gap if gap > 0 else None   # 14B 가 더 좋을 때만 의미
    print(f"  A1-5 macro-F1 14B {res['macro_f1_14b']:.3f} · 32B {res['macro_f1_32b']:.3f} · "
          f"32B→14B 분위수 매핑 {res['macro_f1_32b_quantile_mapped']:.3f}"
          + (f" (격차의 {res['recovered_fraction_of_gap']:.0%} 회복)" if gap > 0 else
             " (32B 가 14B 이상 → 분위수 매핑 실험은 해석 대상 아님)"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--suffix", default="_full")
    ap.add_argument("--out-dir", default="analysis_out/a1")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    ds = load_dataset(args.processed_dir)
    preds = predict_all(ds, args.results_dir, args.suffix, methods=("factor28",))
    if not all(("factor28", m) in preds for m in MODELS):
        raise SystemExit("factor28 14b·32b 결과가 모두 필요합니다.")
    X = {m: preds[("factor28", m)].X for m in MODELS}
    res = {"A1-1": a1_1(ds, X, args.out_dir), "A1-2": a1_2(ds, X, args.out_dir),
           "A1-3": a1_3(ds, X, args.out_dir, max(args.n_boot // 2, 200), rng),
           "A1-4": a1_4(ds, preds, args.out_dir, args.n_boot, rng),
           "A1-5": a1_5(ds, preds, args.out_dir)}
    json.dump(res, open(os.path.join(args.out_dir, "a1_summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"  저장: {args.out_dir}/a1_summary.json")


if __name__ == "__main__":
    main()
