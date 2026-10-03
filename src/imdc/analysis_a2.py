# -*- coding: utf-8 -*-
"""
analysis_a2.py — 심사위원 A-2: 정상군 성능 저하 (A2-1 ~ A2-7). 제안 방법(factor28) 기준.

  python -m imdc.analysis_a2 --results-dir results --suffix _full --out-dir analysis_out/a2

  A2-1 정상 precision/recall, 정상→각 장애 / 각 장애→정상 비율, 장애군 민감도 (+클러스터 CI)
  A2-2 표본 수: 범주별 32명 균등 다운샘플링 × 50회 → 범주별 F1 분포
  A2-3 표현 구조: 요약 특징(활성 요인 수·평균·최대·엔트로피) 범주별 분포·중첩(AUC),
       28차원 + 요약 4개 LR 재학습 → 정상군 F1 변화 (paired 부트스트랩)
  A2-4 희소 라벨: 골드 양성 세션 < 20 인 요인 제거 / 희소 요인만 → 재학습
  A2-4b 희소 라벨 ②: 범주별 골드 라벨 밀도(골드 없는 세션 비율 등) vs LLM 점수 희소성
  A2-5 '언급 부재 ≠ 증상 부재': 장애 내담자의 저신호 세션 비율, 정상으로 오분류된 장애 내담자 특성
  A2-6 2단계 분류: (장애 vs 정상) → (장애 내 3분류) + 3클래스 재학습 macro-F1 (4.1절 0.820/0.794 산출 방식 확인용)
  A2-7 운영점: P(NORMAL) 임계값별 장애군 민감도·특이도 (고정 임계값, 선택 아님 → 낙관 편향 없음)
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

from . import stats as S
from .constants import CLASSES, DISORDERS, MIN_POS_SPARSE, N_BOOT, SEED, SHORT, SYMPTOM28
from .core import load_dataset, make_lr, oof_proba, person_from_proba, predict_all
from .paper import save_csv


def summary_feats(X):
    n_act = (X >= 1).sum(1)
    cnt = np.stack([(X == k).sum(1) for k in range(4)], 1) / X.shape[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(cnt > 0, cnt * np.log2(cnt), 0), 1)
    return np.column_stack([n_act, X.mean(1), X.max(1), ent])


SUMMARY_NAMES = ["n_active", "mean_score", "max_score", "score_entropy"]


def a2_1(ds, mp, n_boot, rng):
    yt, yp = ds.person_y, mp.person_pred
    d = S.summary(yt, yp)
    out = {k: d[k] for k in ("NOR_precision", "NOR_recall", "disorder_sensitivity", "disorder_to_normal")}
    nor = yt == "NORMAL"
    for c in DISORDERS:
        out[f"NOR_to_{SHORT[c]}"] = float((yp[nor] == c).mean())
        out[f"{SHORT[c]}_to_NOR"] = float((yp[yt == c] == "NORMAL").mean())
    for k, fn in [("disorder_sensitivity", lambda ii: S.summary(yt[ii], yp[ii])["disorder_sensitivity"]),
                  ("NOR_recall", lambda ii: S.summary(yt[ii], yp[ii])["NOR_recall"])]:
        lo, hi, _ = S.boot_ci(fn, ds.persons, n_boot, rng)
        out[k + "_ci"] = [lo, hi]
    print(f"  A2-1 [{mp.model}] 정상 P {out['NOR_precision']:.3f} R {out['NOR_recall']:.3f} · "
          f"장애군 민감도 {out['disorder_sensitivity']:.3f} [{out['disorder_sensitivity_ci'][0]:.3f}, "
          f"{out['disorder_sensitivity_ci'][1]:.3f}] · 장애→정상 {out['disorder_to_normal']:.1%}")
    return out


def a2_2(ds, mp, reps, rng):
    by_cls = {c: [p for p in ds.persons if ds.person_label[p] == c] for c in CLASSES}
    n_min = min(len(v) for v in by_cls.values())
    f1s = {c: [] for c in CLASSES}
    for _ in range(reps):
        persons = sorted(sum((list(rng.choice(v, n_min, replace=False)) for v in by_cls.values()), []))
        rows = np.concatenate([ds.sess_of[p] for p in persons])
        P = oof_proba(mp.X, ds, rows=rows)
        yp = person_from_proba(P, ds, persons=persons)
        yt = np.array([ds.person_label[p] for p in persons])
        pc = S.per_class(yt, yp)
        for c in CLASSES:
            f1s[c].append(pc[c]["F1"])
    res = {c: {"mean": float(np.mean(v)), "sd": float(np.std(v))} for c, v in f1s.items()}
    res["n_per_class"] = n_min
    res["normal_still_lowest_rate"] = float(np.mean(
        [min(CLASSES, key=lambda c: f1s[c][r]) == "NORMAL" for r in range(reps)]))
    print(f"  A2-2 [{mp.model}] 균등 {n_min}명×{reps}회: " + ", ".join(
        f"{SHORT[c]} {res[c]['mean']:.3f}±{res[c]['sd']:.3f}" for c in CLASSES)
        + f" · 정상군 최하위 비율 {res['normal_still_lowest_rate']:.0%}")
    return res


def a2_3(ds, mp, out_dir, n_boot, rng):
    Z = summary_feats(mp.X)
    # 내담자 평균 요약 특징의 범주별 분포 + 정상 vs 장애 중첩(AUC; 0.5 = 완전 중첩)
    Zp = np.array([Z[ds.sess_of[p]].mean(0) for p in ds.persons])
    yt = ds.person_y
    rows = [["feature"] + [f"mean_{SHORT[c]}" for c in CLASSES] + ["AUC_disorder_vs_NOR"]]
    overlap = {}
    for j, nm in enumerate(SUMMARY_NAMES):
        auc = roc_auc_score(yt != "NORMAL", Zp[:, j])
        overlap[nm] = float(auc)
        rows.append([nm] + [f"{Zp[yt == c, j].mean():.3f}" for c in CLASSES] + [f"{auc:.3f}"])
    save_csv(rows, os.path.join(out_dir, f"a2_3_summary_features_{mp.model}.csv"))
    # 28 + 4 재학습
    Xaug = np.column_stack([mp.X, Z])
    yp_aug = person_from_proba(oof_proba(Xaug, ds), ds)
    yp0 = mp.person_pred
    f_nor = lambda yp: (lambda ii: S.per_class(yt[ii], yp[ii])["NORMAL"]["F1"])
    lo, hi, p = S.paired_boot_diff(f_nor(yp_aug), f_nor(yp0), ds.persons, n_boot, rng)
    res = {"overlap_auc": overlap,
           "base": {"NOR_F1": S.per_class(yt, yp0)["NORMAL"]["F1"], "macro_f1": S.macro_f1(yt, yp0)},
           "aug": {"NOR_F1": S.per_class(yt, yp_aug)["NORMAL"]["F1"], "macro_f1": S.macro_f1(yt, yp_aug)},
           "delta_NOR_F1_ci": [lo, hi], "p_boot": p}
    print(f"  A2-3 [{mp.model}] 요약특징 중첩 AUC " + ", ".join(f"{k} {v:.2f}" for k, v in overlap.items()))
    print(f"       28→28+4: 정상 F1 {res['base']['NOR_F1']:.3f}→{res['aug']['NOR_F1']:.3f} "
          f"(Δ CI [{lo:+.3f}, {hi:+.3f}]) · macro {res['base']['macro_f1']:.3f}→{res['aug']['macro_f1']:.3f}")
    return res


def a2_4(ds, mp):
    npos = (ds.gold >= 1).sum(0)
    sparse = [i for i in range(28) if npos[i] < MIN_POS_SPARSE]
    dense = [i for i in range(28) if i not in sparse]
    yt = ds.person_y
    res = {"sparse_factors": [SYMPTOM28[i] for i in sparse], "threshold": MIN_POS_SPARSE}
    for name, cols in [("all", list(range(28))), ("without_sparse", dense), ("sparse_only", sparse)]:
        if not cols:
            res[name] = None; continue
        yp = person_from_proba(oof_proba(mp.X[:, cols], ds), ds)
        res[name] = {"NOR_F1": S.per_class(yt, yp)["NORMAL"]["F1"], "macro_f1": S.macro_f1(yt, yp)}
    print(f"  A2-4 [{mp.model}] 희소 요인 {len(sparse)}개 (골드 양성<{MIN_POS_SPARSE}): "
          + " · ".join(f"{k} NOR {v['NOR_F1']:.3f}/macro {v['macro_f1']:.3f}"
                       for k, v in res.items() if isinstance(v, dict)))
    return res


def a2_4b(ds, mp, out_dir):
    """희소 라벨 ② — 범주별 골드 라벨 밀도와 LLM 점수 희소성.
    '정상군 세션은 전문가 골드 자체가 거의 0이라 학습·검증 신호가 없다'는 해석에 대한 직접 근거.
      - 세션당 골드 양성(≥1) 요인 수, 골드가 하나도 없는 세션 비율 (세션 라벨 / 내담자 라벨 기준)
      - 같은 세션에서 LLM 점수 ≥1 요인 수, LLM 이 전부 0을 준 세션 비율
      - 골드가 없는 세션에서 LLM 이 준 점수 (골드 부재 = 증상 부재인지, 주석 누락인지 판단 재료)
      - 범주별·요인별 골드 양성 세션 수 (정상군 내부에서 희소한 요인 파악)"""
    G = ds.gold >= 1
    L = mp.X >= 1
    plab = np.array([ds.person_label[p] for p in ds.pid])
    rows = [["group_by", "class", "n_sessions", "gold_pos_per_session", "gold_empty_rate",
             "llm_pos_per_session", "llm_all_zero_rate", "llm_pos_in_gold_empty_sessions"]]
    res = {}
    for key, lab in (("session_label", ds.y), ("person_label", plab)):
        res[key] = {}
        for c in CLASSES:
            m = lab == c
            if not m.any():
                continue
            g_empty = ~G[m].any(1)
            d = {"n": int(m.sum()), "gold_pos_per_session": float(G[m].sum(1).mean()),
                 "gold_empty_rate": float(g_empty.mean()),
                 "llm_pos_per_session": float(L[m].sum(1).mean()),
                 "llm_all_zero_rate": float((~L[m].any(1)).mean()),
                 "llm_pos_in_gold_empty": float(L[m][g_empty].sum(1).mean()) if g_empty.any() else None}
            res[key][c] = d
            rows.append([key, c, d["n"], f"{d['gold_pos_per_session']:.3f}", f"{d['gold_empty_rate']:.3f}",
                         f"{d['llm_pos_per_session']:.3f}", f"{d['llm_all_zero_rate']:.3f}",
                         "" if d["llm_pos_in_gold_empty"] is None else f"{d['llm_pos_in_gold_empty']:.3f}"])
    save_csv(rows, os.path.join(out_dir, f"a2_4b_label_density_{mp.model}.csv"))
    frows = [["factor"] + [f"gold_pos_sessions_{SHORT[c]}" for c in CLASSES]]
    for fi, f in enumerate(SYMPTOM28):
        frows.append([f] + [int(G[ds.y == c, fi].sum()) for c in CLASSES])
    save_csv(frows, os.path.join(out_dir, "a2_4b_gold_pos_by_class.csv"))
    d = res["session_label"]
    print(f"  A2-4b [{mp.model}] 골드 없는 세션 비율: " + ", ".join(
        f"{SHORT[c]} {d[c]['gold_empty_rate']:.1%}" for c in CLASSES if c in d)
        + " · 세션당 골드 양성 요인: " + ", ".join(
        f"{SHORT[c]} {d[c]['gold_pos_per_session']:.2f}" for c in CLASSES if c in d))
    print("        LLM 전부 0 세션 비율: " + ", ".join(
        f"{SHORT[c]} {d[c]['llm_all_zero_rate']:.1%}" for c in CLASSES if c in d)
        + " · 골드 없는 세션에서 LLM 양성 요인 수: " + ", ".join(
        f"{SHORT[c]} {d[c]['llm_pos_in_gold_empty']:.2f}" for c in CLASSES
        if c in d and d[c]['llm_pos_in_gold_empty'] is not None))
    return res


def a2_5(ds, mp):
    yt, yp = ds.person_y, mp.person_pred
    n_act = (mp.X >= 1).sum(1)
    dis_sess = np.isin(ds.pid, [p for p in ds.persons if ds.person_label[p] != "NORMAL"])
    low = float((n_act[dis_sess] <= 1).mean())
    pm = {p: (mp.X[ds.sess_of[p]].mean(), n_act[ds.sess_of[p]].mean()) for p in ds.persons}
    dis = yt != "NORMAL"
    miss = [p for p, t, q in zip(ds.persons, yt, yp) if t != "NORMAL" and q == "NORMAL"]
    hit = [p for p, t, q in zip(ds.persons, yt, yp) if t != "NORMAL" and q == t]
    res = {"disorder_sessions_low_signal_rate": low, "n_missed_to_normal": len(miss), "n_correct": len(hit)}
    for j, nm in enumerate(["mean_score", "n_active"]):
        a, b = [pm[p][j] for p in miss], [pm[p][j] for p in hit]
        res[nm] = {"missed": float(np.mean(a)) if a else None, "correct": float(np.mean(b)) if b else None,
                   "normal_persons": float(np.mean([pm[p][j] for p in np.array(ds.persons)[~dis]])),
                   "mannwhitney_p": float(mannwhitneyu(a, b).pvalue) if a and b else None}
    print(f"  A2-5 [{mp.model}] 장애 세션 중 활성요인≤1 비율 {low:.1%} · 정상으로 오분류된 장애 {len(miss)}명: "
          "평균점수 " + " vs ".join(f"{k} {v:.3f}" for k, v in [("오분류", res['mean_score']['missed']),
                                                              ("정답", res['mean_score']['correct']),
                                                              ("정상군", res['mean_score']['normal_persons'])]
                                     if v is not None)
          + (f" (Mann-Whitney p={res['mean_score']['mannwhitney_p']:.3g})"
             if res['mean_score']['mannwhitney_p'] is not None else ""))
    return res


def a2_6(ds, mp):
    """2단계: ① 세션 LR(장애 vs 정상) → 내담자 soft vote ② 장애로 판정된 내담자는 3분류 LR(장애 세션만 학습)."""
    yt = ds.person_y
    is_nor = (ds.y == "NORMAL").astype(int)
    p_nor = np.zeros(len(ds.sids)); P3 = np.zeros((len(ds.sids), 3))
    for f in sorted(set(ds.fold.tolist())):
        tr, te = np.where(ds.fold != f)[0], np.where(ds.fold == f)[0]
        m1 = make_lr().fit(mp.X[tr], is_nor[tr])
        p_nor[te] = m1.predict_proba(mp.X[te])[:, list(m1.classes_).index(1)]
        tr3 = tr[ds.y[tr] != "NORMAL"]
        m2 = make_lr().fit(mp.X[tr3], ds.y[tr3]); cls = list(m2.classes_)
        P3[te] = m2.predict_proba(mp.X[te])[:, [cls.index(c) for c in DISORDERS]]
    yp = []
    for p in ds.persons:
        ii = ds.sess_of[p]
        yp.append("NORMAL" if p_nor[ii].mean() >= 0.5 else DISORDERS[int(P3[ii].mean(0).argmax())])
    yp = np.array(yp)
    # 3클래스 재학습: 정상 내담자를 제외하고 장애 라벨 세션으로만 학습, 장애 내담자만 평가
    pers3 = [p for p in ds.persons if ds.person_label[p] != "NORMAL"]
    in3 = np.isin(ds.pid, pers3)
    P = np.zeros((len(ds.sids), len(CLASSES)))
    for f in sorted(set(ds.fold.tolist())):
        tr = np.where(in3 & (ds.fold != f) & (ds.y != "NORMAL"))[0]
        te = np.where(in3 & (ds.fold == f))[0]
        m = make_lr().fit(mp.X[tr], ds.y[tr]); cls = list(m.classes_)
        P[te] = np.column_stack([m.predict_proba(mp.X[te])[:, cls.index(c)] if c in cls
                                 else np.zeros(len(te)) for c in CLASSES])
    yp3 = person_from_proba(P, ds, persons=pers3)
    yt3 = np.array([ds.person_label[p] for p in pers3])
    res = {"two_stage_4cls_macro_f1": S.macro_f1(yt, yp), "two_stage_NOR_F1": S.per_class(yt, yp)["NORMAL"]["F1"],
           "two_stage_disorder_sensitivity": S.summary(yt, yp)["disorder_sensitivity"],
           "one_stage_4cls_macro_f1": S.macro_f1(yt, mp.person_pred),
           "macro_f1_3cls_posthoc": S.summary(yt, mp.person_pred)["macro_f1_3cls_posthoc"],
           "macro_f1_3cls_retrained": S.macro_f1(yt3, yp3, labels=DISORDERS)}
    print(f"  A2-6 [{mp.model}] 2단계 macro {res['two_stage_4cls_macro_f1']:.3f} (1단계 {res['one_stage_4cls_macro_f1']:.3f}) "
          f"· 정상 F1 {res['two_stage_NOR_F1']:.3f} · 3클래스: 사후 {res['macro_f1_3cls_posthoc']:.3f} / "
          f"재학습 {res['macro_f1_3cls_retrained']:.3f}")
    return res


def a2_7(ds, mp):
    yt = ds.person_y
    pn = np.array([mp.sess_proba[ds.sess_of[p], CLASSES.index("NORMAL")].mean() for p in ds.persons])
    dis = yt != "NORMAL"
    res = {"auc_disorder_vs_normal": float(roc_auc_score(dis, -pn)), "points": []}
    for t in (0.20, 0.25, 0.30, 0.35, 0.40, 0.50):
        pred_dis = pn < t
        res["points"].append({"threshold_P_NOR": t, "sensitivity": float((pred_dis & dis).sum() / dis.sum()),
                              "specificity": float((~pred_dis & ~dis).sum() / (~dis).sum())})
    argmax_dis = mp.person_pred != "NORMAL"
    res["argmax"] = {"sensitivity": float((argmax_dis & dis).sum() / dis.sum()),
                     "specificity": float((~argmax_dis & ~dis).sum() / (~dis).sum())}
    print(f"  A2-7 [{mp.model}] AUC {res['auc_disorder_vs_normal']:.3f} · argmax 민감도/특이도 "
          f"{res['argmax']['sensitivity']:.3f}/{res['argmax']['specificity']:.3f} · " + ", ".join(
              f"t={d['threshold_P_NOR']:.2f}: {d['sensitivity']:.3f}/{d['specificity']:.3f}" for d in res["points"]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--suffix", default="_full")
    ap.add_argument("--out-dir", default="analysis_out/a2")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    ds = load_dataset(args.processed_dir)
    preds = predict_all(ds, args.results_dir, args.suffix, methods=("factor28",))
    res = {}
    for model in ("14b", "32b"):
        mp = preds.get(("factor28", model))
        if mp is None:
            continue
        rng = np.random.default_rng(args.seed)
        print(f"\n── {model.upper()} ──")
        res[model] = {"A2-1": a2_1(ds, mp, args.n_boot, rng), "A2-2": a2_2(ds, mp, args.reps, rng),
                      "A2-3": a2_3(ds, mp, args.out_dir, args.n_boot, rng), "A2-4": a2_4(ds, mp),
                      "A2-4b": a2_4b(ds, mp, args.out_dir),
                      "A2-5": a2_5(ds, mp), "A2-6": a2_6(ds, mp), "A2-7": a2_7(ds, mp)}
    json.dump(res, open(os.path.join(args.out_dir, "a2_summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"\n  저장: {args.out_dir}/a2_summary.json")


if __name__ == "__main__":
    main()
