# -*- coding: utf-8 -*-
"""
paper.py — 결과 파일에서 논문 Table 3·4, Fig. 2·3 을 한 번에 재생성.

  python -m imdc.paper --results-dir results --suffix _full --out-dir paper_out

산출물 (out-dir)
  table3_performance.csv     내담자 단위 macro-F1 [95% 클러스터 부트스트랩 CI], acc, 범주별 F1,
                             + 정상 precision/recall, 장애군 민감도 (A2-1 행), 파싱 실패율
  table3_confusion.json      8개 구성의 내담자 혼동행렬 (행=실제, 열=예측, 순서 ADD/ANX/DEP/NOR)
  table4_faithfulness.csv    요인별 n(골드≥1), n(골드≥2), Spearman ρ, QWK, recall(≥2) — 14B·32B
  fig2_confusion.png         제안 방법 14B·32B 혼동행렬 (+ 캡션용 오분류율 출력)
  fig3_coef_{14b,32b}.png    표준화 계수 (그룹 부트스트랩, BH-FDR) — Normal 패널은 음(−) 계수
  fig3_coef_{14b,32b}.csv    28요인 × 4범주 계수 평균·95% CI·q
"""
import argparse
import csv
import json
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from . import stats as S
from .constants import (CLASSES, METHOD_LABEL, MIN_POS_TABLE4, N_BOOT, N_BOOT_COEF, SEED, SHORT,
                        SYMPTOM28)
from .core import load_dataset, make_lr, predict_all

TABLE3_ORDER = ["zeroshot", "fewshot", "direct4", "factor28"]


def save_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(rows)
    print(f"  저장: {path}")


# ── Table 3 ─────────────────────────────────────────────────────
def table3(ds, preds, out_dir, n_boot, seed):
    rng = np.random.default_rng(seed)
    yt = ds.person_y
    head = ["Model", "Method", "macro_F1", "CI_low", "CI_high", "Acc"] + \
           [f"F1_{SHORT[c]}" for c in CLASSES] + \
           ["NOR_precision", "NOR_recall", "Disorder_sensitivity", "Disorder_to_NOR",
            "macroF1_3cls_posthoc", "parse_fail_rate", "vote_ties"]
    rows, cms = [head], {}
    for model in ("14b", "32b"):
        for method in TABLE3_ORDER:
            mp = preds.get((method, model))
            if mp is None:
                continue
            yp = mp.person_pred
            d = S.summary(yt, yp)
            lo, hi, _ = S.boot_ci(lambda ii: S.macro_f1(yt[ii], yp[ii]), ds.persons, n_boot, rng)
            rows.append([model.upper(), METHOD_LABEL[method], f"{d['macro_f1']:.3f}", f"{lo:.3f}",
                         f"{hi:.3f}", f"{d['acc']:.3f}"] +
                        [f"{d['F1_' + c]:.3f}" for c in CLASSES] +
                        [f"{d['NOR_precision']:.3f}", f"{d['NOR_recall']:.3f}",
                         f"{d['disorder_sensitivity']:.3f}", f"{d['disorder_to_normal']:.3f}",
                         f"{d['macro_f1_3cls_posthoc']:.3f}", f"{mp.parse_fail:.3f}", mp.ties])
            cms[f"{method}_{model}"] = confusion_matrix(yt, yp, labels=CLASSES).tolist()
    w = max(len(h) for h in head[:6])
    for r in rows:
        print("  " + " | ".join(str(x)[:24].rjust(8) for x in r[:10]))
    save_csv(rows, os.path.join(out_dir, "table3_performance.csv"))
    json.dump({"order": CLASSES, "rows=true, cols=pred": cms},
              open(os.path.join(out_dir, "table3_confusion.json"), "w"), indent=1)
    return cms


# ── Table 4 ─────────────────────────────────────────────────────
def faithfulness(g, l):
    rho = spearmanr(g, l).correlation if len(set(g)) > 1 and len(set(l)) > 1 else float("nan")
    try:
        qwk = cohen_kappa_score(g, l, weights="quadratic", labels=[0, 1, 2, 3])
    except Exception:
        qwk = float("nan")
    m = g >= 2
    rec = float((l[m] >= 2).mean()) if m.any() else float("nan")
    return rho, qwk, rec


def table4(ds, preds, out_dir, min_pos):
    head = ["factor", "n_gold_ge1", "n_gold_ge2"]
    models = [m for m in ("14b", "32b") if ("factor28", m) in preds]
    for k in ("rho", "qwk", "recall"):
        head += [f"{k}_{m}" for m in models]
    head.append("verified")
    body = []
    for fi, f in enumerate(SYMPTOM28):
        g = ds.gold[:, fi]
        row = [f, int((g >= 1).sum()), int((g >= 2).sum())]
        vals = {m: faithfulness(g, preds[("factor28", m)].X[:, fi]) for m in models}
        for k in range(3):
            row += [f"{vals[m][k]:.3f}" for m in models]
        row.append("Y" if row[2] >= min_pos else "N")
        body.append(row)
    key = 3  # rho of first model
    body.sort(key=lambda r: (r[-1] != "Y", -float(r[key]) if r[key] != "nan" else 0))
    ver = [r for r in body if r[-1] == "Y"]
    mean_row = ["MEAN(verified)", "", ""]
    for j in range(3, 3 + 3 * len(models)):
        v = [float(r[j]) for r in ver if r[j] != "nan"]
        mean_row.append(f"{np.mean(v):.3f}" if v else "nan")
    mean_row.append(f"{len(ver)} factors")
    save_csv([head] + body + [mean_row], os.path.join(out_dir, "table4_faithfulness.csv"))
    print(f"  검증 대상 {len(ver)}개 · 평균 " + " · ".join(f"{h}={v}" for h, v in zip(head[3:], mean_row[3:])))


# ── Fig. 2 ──────────────────────────────────────────────────────
def fig2(ds, preds, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = [m for m in ("14b", "32b") if ("factor28", m) in preds]
    if not models:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 4.6), squeeze=False)
    yt = ds.person_y
    for ax, model, tag in zip(axes[0], models, ["(a)", "(b)"]):
        cm = confusion_matrix(yt, preds[("factor28", model)].person_pred, labels=CLASSES)
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
        ax.set_xticks(range(4)); ax.set_xticklabels([SHORT[c] for c in CLASSES])
        ax.set_yticks(range(4)); ax.set_yticklabels([SHORT[c] for c in CLASSES])
        ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
        ax.set_title(f"{tag} {model.upper()}", fontsize=11)
        thr = cm.max() / 2
        for i in range(4):
            for j in range(4):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=11,
                        color="white" if cm[i, j] > thr else "black")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        # 캡션용 수치 (A2-1)
        nor = CLASSES.index("NORMAL")
        rowN = cm[nor]
        print(f"  [{model}] 정상→" + " / ".join(f"{SHORT[c]} {rowN[j] / rowN.sum():.1%}"
                                              for j, c in enumerate(CLASSES) if c != "NORMAL")
              + " · " + " / ".join(f"{SHORT[c]}→정상 {cm[i, nor] / cm[i].sum():.1%}"
                                   for i, c in enumerate(CLASSES) if c != "NORMAL"))
    plt.tight_layout()
    p = os.path.join(out_dir, "fig2_confusion.png")
    plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  저장: {p}")


# ── Fig. 3 ──────────────────────────────────────────────────────
CLASS_COLOR = {"ADDICTION": "#E36F87", "ANXIETY": "#9CC0E4", "DEPRESSION": "#F0B761",
               "NORMAL": "#8FCBA1"}


def coef_bootstrap(X, ds, n_boot, rng):
    """그룹(내담자) 부트스트랩으로 전체 적합 LR 의 표준화 계수 분포 (n_boot, 4, 28)."""
    B = np.zeros((n_boot, len(CLASSES), X.shape[1]))
    for b, ii in enumerate(S.cluster_boot_samples(ds.pid, n_boot, rng)):
        m = make_lr().fit(X[ii], ds.y[ii])
        cls = list(m.classes_)
        B[b] = m.named_steps["logisticregression"].coef_[[cls.index(c) for c in CLASSES]]
    frac = (B > 0).mean(0)
    p = 2 * np.minimum(frac, 1 - frac)
    q = S.bh(p.ravel()).reshape(p.shape)
    return B, q


def fig3(ds, preds, out_dir, n_boot, seed, top, normal_mode):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for model in ("14b", "32b"):
        mp = preds.get(("factor28", model))
        if mp is None:
            continue
        rng = np.random.default_rng(seed)
        B, q = coef_bootstrap(mp.X, ds, n_boot, rng)
        mean = B.mean(0)
        lo, hi = np.percentile(B, [2.5, 97.5], axis=0)
        rows = [["class", "factor", "coef_mean", "ci_low", "ci_high", "q_bh", "significant"]]
        for ci, c in enumerate(CLASSES):
            for fi, f in enumerate(SYMPTOM28):
                sig = q[ci, fi] < 0.05 and (lo[ci, fi] > 0 or hi[ci, fi] < 0)
                rows.append([c, f, f"{mean[ci, fi]:+.4f}", f"{lo[ci, fi]:+.4f}", f"{hi[ci, fi]:+.4f}",
                             f"{q[ci, fi]:.4g}", int(sig)])
        save_csv(rows, os.path.join(out_dir, f"fig3_coef_{model}.csv"))

        fig, ax = plt.subplots(figsize=(8.5, 3.9))
        ax.set_axisbelow(True); ax.grid(axis="y", color="#DBDBDB", linewidth=0.8)
        xpos, xt, xl = 0, [], []
        for ci, c in enumerate(CLASSES):
            neg = (c == "NORMAL" and normal_mode == "negative")
            order = np.argsort(mean[ci]) if neg else np.argsort(-mean[ci])
            idx = [i for i in order if (mean[ci][i] < 0 if neg else mean[ci][i] > 0)][:top]
            for i in idx:
                sig = q[ci][i] < 0.05
                ax.bar(xpos, mean[ci][i], color=CLASS_COLOR[c], width=0.82, zorder=3,
                       edgecolor="black" if sig else "none", linewidth=1.4 if sig else 0,
                       alpha=1.0 if sig else 0.55)
                xt.append(xpos); xl.append(SYMPTOM28[i]); xpos += 1
            xpos += 0.9
        ax.axhline(0, color="#555", linewidth=0.8)
        ax.set_xticks(xt); ax.set_xticklabels(xl, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("Standardized coefficient", fontsize=10)
        ax.margins(x=0.01)
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=CLASS_COLOR[c], edgecolor="#C8C8C8",
                                 linewidth=0.6) for c in CLASSES]
        names = [c.title() + (" (−)" if c == "NORMAL" and normal_mode == "negative" else "")
                 for c in CLASSES]
        ax.legend(handles, names, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                  frameon=False, fontsize=9.5, handlelength=1.5, handleheight=1.0)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        p = os.path.join(out_dir, f"fig3_coef_{model}.png")
        plt.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
        print(f"  저장: {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--suffix", default="_full")
    ap.add_argument("--out-dir", default="paper_out")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--n-boot-coef", type=int, default=N_BOOT_COEF)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--normal-mode", choices=["negative", "positive"], default="negative",
                    help="Fig. 3 Normal 패널: 음(−) 계수 상위(권장, A-2 대응) / 양(+) 계수 상위")
    ap.add_argument("--min-pos", type=int, default=MIN_POS_TABLE4)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ds = load_dataset(args.processed_dir)
    print(f"[데이터] 세션 {len(ds.sids)} · 내담자 {len(ds.persons)}")
    preds = predict_all(ds, args.results_dir, args.suffix)
    print("\n[Table 3]"); table3(ds, preds, args.out_dir, args.n_boot, args.seed)
    print("\n[Table 4]"); table4(ds, preds, args.out_dir, args.min_pos)
    print("\n[Fig. 2]"); fig2(ds, preds, args.out_dir)
    print("\n[Fig. 3]"); fig3(ds, preds, args.out_dir, args.n_boot_coef, args.seed, args.top,
                              args.normal_mode)


if __name__ == "__main__":
    main()
