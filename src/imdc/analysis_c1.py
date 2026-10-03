# -*- coding: utf-8 -*-
"""
analysis_c1.py — 심사위원 C-(1): 통계 해석 · 희소 골드 · 베이스라인 공정성 · 세션/내담자 정합성.

  python -m imdc.analysis_c1 --results-dir results --suffix _full --out-dir analysis_out/c1

  C1-1 제안 vs 각 베이스라인: 내담자 McNemar + Holm(모델별 3개 비교가 한 검정 가족),
       ΔmacroF1 paired 클러스터 부트스트랩 CI
  C1-2 Table 4 확장: 요인별 n, ρ 의 클러스터 부트스트랩 CI·p → BH-FDR q,
       희소 요인(골드≥1 세션 < 20) 제외 민감도 분석
  C1-3 베이스라인 공정성 점검: 8개 결과 파일이 세션별로 **같은 전사(sha1)** 를 입력받았는지,
       입력 토큰 수, 생성 길이 한도 도달, 파싱 실패율, few-shot 예시 구성
  C1-4 세션 단위 성능(클러스터 CI) + 집계 규칙 비교(soft / hard / max) + 세션·내담자 라벨 불일치
"""
import argparse
import json
import os
from collections import Counter

import numpy as np
from scipy.stats import spearmanr

from . import stats as S
from .constants import (CLASSES, DIRECT_METHODS, LR_METHODS, METHOD_LABEL, METHODS, MIN_POS_SPARSE,
                        MIN_POS_TABLE4, N_BOOT, SEED, SYMPTOM28)
from .core import load_dataset, normalize_sid, person_from_proba, predict_all, read_jsonl, result_path
from .paper import save_csv
from .prompts import FEWSHOT_EXAMPLES


def c1_1(ds, preds, n_boot, rng):
    yt = ds.person_y
    res = {}
    for model in ("14b", "32b"):
        fac = preds.get(("factor28", model))
        if fac is None:
            continue
        raw, info = {}, {}
        for b in ("zeroshot", "fewshot", "direct4"):
            mp = preds.get((b, model))
            if mp is None:
                continue
            mc = S.mcnemar(fac.person_pred == yt, mp.person_pred == yt)
            lo, hi, pb = S.paired_boot_diff(lambda ii: S.macro_f1(yt[ii], fac.person_pred[ii]),
                                            lambda ii: S.macro_f1(yt[ii], mp.person_pred[ii]),
                                            ds.persons, n_boot, rng)
            raw[b] = mc["p"]
            info[b] = {"delta_macro_f1": S.macro_f1(yt, fac.person_pred) - S.macro_f1(yt, mp.person_pred),
                       "ci": [lo, hi], "p_boot": pb, "mcnemar": mc}
        for b, ph in S.holm(raw).items():
            info[b]["p_holm"] = ph
        res[model] = info
        for b, d in info.items():
            print(f"  C1-1 [{model}] 제안 vs {METHOD_LABEL[b]:12}: ΔF1 {d['delta_macro_f1']:+.3f} "
                  f"[{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}] · McNemar p={d['mcnemar']['p']:.2e} "
                  f"→ Holm {d['p_holm']:.2e}")
    return res


def c1_2(ds, preds, out_dir, n_boot, rng):
    npos1, npos2 = (ds.gold >= 1).sum(0), (ds.gold >= 2).sum(0)
    res = {}
    rows = [["factor", "n_gold_ge1", "n_gold_ge2"]]
    models = [m for m in ("14b", "32b") if ("factor28", m) in preds]
    for m in models:
        rows[0] += [f"rho_{m}", f"rho_ci_low_{m}", f"rho_ci_high_{m}", f"p_boot_{m}", f"q_bh_{m}"]
    table = {f: [f, int(npos1[i]), int(npos2[i])] for i, f in enumerate(SYMPTOM28)}
    for m in models:
        X = preds[("factor28", m)].X
        rho, ci, pb = [], [], []
        for fi in range(28):
            g, l = ds.gold[:, fi], X[:, fi]

            def r(ii, g=g, l=l):
                if len(set(g[ii])) < 2 or len(set(l[ii])) < 2:
                    return np.nan
                return spearmanr(g[ii], l[ii]).correlation
            est = r(np.arange(len(g)))
            lo, hi, vals = S.boot_ci(r, ds.pid, n_boot, rng)
            v = vals[~np.isnan(vals)]
            p = 2 * min((v <= 0).mean(), (v >= 0).mean()) if len(v) else np.nan
            rho.append(est); ci.append((lo, hi)); pb.append(min(p, 1.0))
        pb = np.array(pb, float)
        q = np.full(28, np.nan)
        ok = ~np.isnan(pb)
        q[ok] = S.bh(pb[ok])
        for i, f in enumerate(SYMPTOM28):
            table[f] += [f"{rho[i]:.3f}", f"{ci[i][0]:.3f}", f"{ci[i][1]:.3f}", f"{pb[i]:.4f}", f"{q[i]:.4f}"]
        rho = np.array(rho, float)
        res[m] = {"mean_rho_all": float(np.nanmean(rho)),
                  f"mean_rho_ge1_ge{MIN_POS_SPARSE}": float(np.nanmean(rho[npos1 >= MIN_POS_SPARSE])),
                  f"mean_rho_verified_ge2_ge{MIN_POS_TABLE4}": float(np.nanmean(rho[npos2 >= MIN_POS_TABLE4])),
                  "n_sig_q05": int(np.nansum(q < 0.05))}
        print(f"  C1-2 [{m}] 평균 ρ: 전체 {res[m]['mean_rho_all']:.3f} · 골드≥1이 {MIN_POS_SPARSE}세션 이상 "
              f"{res[m][f'mean_rho_ge1_ge{MIN_POS_SPARSE}']:.3f} · 검증대상 "
              f"{res[m][f'mean_rho_verified_ge2_ge{MIN_POS_TABLE4}']:.3f} · q<0.05 요인 {res[m]['n_sig_q05']}개")
    save_csv(rows + list(table.values()), os.path.join(out_dir, "c1_2_table4_extended.csv"))
    return res


def c1_3(ds, results_dir, suffix):
    """8개 파일이 같은 전사를 받았는지 + 입력/출력 조건 점검 (신규 _full 결과에서만 완전 동작)."""
    res = {"files": {}}
    sha_by_file = {}
    for model in ("14b", "32b"):
        for method in METHODS:
            path = result_path(results_dir, method, model, suffix)
            if not os.path.exists(path):
                continue
            recs = {normalize_sid(r["session_id"]): r for r in read_jsonl(path)}
            sha_by_file[f"{method}_{model}"] = {s: r.get("transcript_sha1") for s, r in recs.items()}
            ntok = [r.get("n_prompt_tokens") for r in recs.values() if r.get("n_prompt_tokens")]
            st = Counter(r.get("parse_status", "legacy") for r in recs.values())
            res["files"][f"{method}_{model}"] = {
                "n": len(recs), "parse_status": dict(st),
                "finish_length": sum(r.get("finish_reason") == "length" for r in recs.values()),
                "prompt_tokens_mean": round(float(np.mean(ntok)), 1) if ntok else None,
                "prompt_tokens_max": int(max(ntok)) if ntok else None}
    keys = list(sha_by_file)
    if keys and all(sha_by_file[k] and None not in sha_by_file[k].values() for k in keys):
        ref = sha_by_file[keys[0]]
        mism = {k: sum(sha_by_file[k].get(s) != ref[s] for s in ref) for k in keys}
        res["transcript_identical_across_files"] = all(v == 0 for v in mism.values())
        res["transcript_mismatch_counts"] = mism
    else:
        res["transcript_identical_across_files"] = None   # 전사 해시가 없는 결과 형식
    res["fewshot_design"] = {
        "n_demonstrations": len(FEWSHOT_EXAMPLES), "per_class": 1,
        "source": "researcher-written synthetic utterances (not drawn from the dataset → no fold leakage)",
        "sentences_per_demo": {c: len(v) for c, v in FEWSHOT_EXAMPLES.items()},
        "chars_per_demo": {c: len("\n".join(v)) for c, v in FEWSHOT_EXAMPLES.items()}}
    for k, v in res["files"].items():
        print(f"  C1-3 {k:22} n={v['n']} 파싱 {v['parse_status']} 생성한도 {v['finish_length']} "
              f"입력토큰 평균/최대 {v['prompt_tokens_mean']}/{v['prompt_tokens_max']}")
    print(f"       세션별 전사 동일성(8개 파일): {res['transcript_identical_across_files']}")
    return res


def c1_4(ds, preds, n_boot, rng):
    res = {}
    yt_s = ds.y
    for (method, model), mp in sorted(preds.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        sp = mp.sess_pred
        lo, hi, _ = S.boot_ci(lambda ii: S.macro_f1(yt_s[ii], sp[ii]), ds.pid, n_boot, rng)
        d = {"session_macro_f1": S.macro_f1(yt_s, sp), "session_ci": [lo, hi],
             "session_acc": float((yt_s == sp).mean()),
             "person_macro_f1": S.macro_f1(ds.person_y, mp.person_pred)}
        if method in LR_METHODS:
            for rule in ("soft", "hard", "max"):
                d[f"person_macro_f1_{rule}"] = S.macro_f1(ds.person_y, person_from_proba(mp.sess_proba, ds, rule))
        res[f"{method}_{model}"] = d
        extra = (" · 집계 soft/hard/max " + "/".join(f"{d[f'person_macro_f1_{r}']:.3f}" for r in ("soft", "hard", "max"))
                 if method in LR_METHODS else "")
        print(f"  C1-4 {method:9}{model}: 세션 macro-F1 {d['session_macro_f1']:.3f} [{lo:.3f}, {hi:.3f}] "
              f"· 내담자 {d['person_macro_f1']:.3f}{extra}")
    # 세션 라벨 vs 내담자 라벨
    plab = np.array([ds.person_label[p] for p in ds.pid])
    dis = plab != "NORMAL"
    res["label_consistency"] = {
        "disorder_person_sessions": int(dis.sum()),
        "of_which_session_label_NORMAL": int((dis & (ds.y == "NORMAL")).sum()),
        "persons_with_mixed_labels": int(sum(len(set(ds.y[ds.sess_of[p]])) > 1 for p in ds.persons))}
    lc = res["label_consistency"]
    print(f"  C1-4 장애 내담자 세션 {lc['disorder_person_sessions']}개 중 세션 라벨 NORMAL "
          f"{lc['of_which_session_label_NORMAL']}개 · 라벨 혼재 내담자 {lc['persons_with_mixed_labels']}명")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--suffix", default="_full")
    ap.add_argument("--out-dir", default="analysis_out/c1")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    ds = load_dataset(args.processed_dir)
    preds = predict_all(ds, args.results_dir, args.suffix)
    res = {"C1-1": c1_1(ds, preds, args.n_boot, rng),
           "C1-2": c1_2(ds, preds, args.out_dir, max(args.n_boot // 2, 200), rng),
           "C1-3": c1_3(ds, args.results_dir, args.suffix),
           "C1-4": c1_4(ds, preds, args.n_boot, rng)}
    json.dump(res, open(os.path.join(args.out_dir, "c1_summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"  저장: {args.out_dir}/c1_summary.json")


if __name__ == "__main__":
    main()
