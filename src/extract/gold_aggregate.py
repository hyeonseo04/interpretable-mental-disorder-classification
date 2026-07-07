#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_coefficients.py — Fig.3: 클래스별 기여 증상요인의 표준화 계수 막대그래프

  다항 로지스틱 회귀(GroupKFold 학습 아님, 전체 적합)의 표준화 계수를
  그룹(사람) 부트스트랩으로 재추정하고, BH-FDR 보정 후 유의한 요인을 강조한다.
  클래스별로 계수 상위 요인을 막대로 표시(양의 기여).

  데이터 규약: session_id 정규화, person_id 대문자.
  (계수·유의성 산출 로직은 stats.py 와 동일 시드/방식 — 값 일치)

  ※ 그림 스타일: 유의(q<0.05)한 요인은 검은 테두리+진한 색, 비유의 요인은 흐리게 표시,
    가로 격자선, 상단 가로 범례(Title Case, 네모 마커).

사용:
  uv run --with scikit-learn --with scipy --with matplotlib plot_coefficients.py \
      --input ../results/session_llm_14b.jsonl --out ../figures/fig3_coef_14b.png --top 5
"""

import argparse
import json
import re

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

SYMPTOM28 = [
    "depressive_mood", "worthlessness", "guilt", "impaired_cognition", "suicidal", "anhedonia",
    "psychomotor_changes", "weight_appetite", "sleep_disturbance", "fatigue",
    "anxiety_mood", "derealization", "perceived_loss_of_control", "anxiety_control",
    "concentration", "avoidance", "physical_symptoms", "irritability",
    "loss_of_control", "craving", "lying", "tolerance", "withdrawal", "salience",
    "resource_investment", "daily_functioning", "social_problems", "negative_consequences",
]
CLASS_ORDER = ["ADDICTION", "ANXIETY", "DEPRESSION", "NORMAL"]
# 두 번째 그림 색감에 맞춰 살짝 더 선명하게 (원하면 자유롭게 조정)
CLASS_COLOR = {"ADDICTION": "#E36F87", "ANXIETY": "#9CC0E4",
               "DEPRESSION": "#F0B761", "NORMAL": "#8FCBA1"}


def norm(s):
    s = str(s).strip().replace("_raw_", "_check_")
    return re.sub(r"(_check_)([a-z])", lambda m: m.group(1) + m.group(2).upper(), s)


def load(path):
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
        r["person_id"] = str(r["person_id"]).upper()
        rows.append(r)
    return rows


def bh(pvals):
    p = np.array(pvals); order = p.argsort(); m = len(p); r = np.empty(m)
    for rank, i in enumerate(order, 1):
        r[i] = p[i] * m / rank
    s = r[order][::-1]; s = np.minimum.accumulate(s)[::-1]
    out = np.empty(m); out[order] = s
    return np.clip(out, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="figures/fig3_coef.png")
    ap.add_argument("--top", type=int, default=6, help="클래스별 상위 요인 수")
    ap.add_argument("--nboot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = load(args.input)
    X = np.array([[r["llm"][f] for f in SYMPTOM28] for r in rows], float)
    y = np.array([r["class"] for r in rows])
    grp = np.array([r["person_id"] for r in rows])
    labels = [c for c in CLASS_ORDER if c in set(y)]

    def pipe():
        return make_pipeline(StandardScaler(),
                             LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))

    # 그룹 부트스트랩 계수
    uniq = list(set(grp)); gidx = {g: np.where(grp == g)[0] for g in uniq}
    B = np.zeros((args.nboot, len(labels), len(SYMPTOM28)))
    for b in range(args.nboot):
        sg = rng.choice(len(uniq), len(uniq), replace=True)
        ii = np.concatenate([gidx[uniq[k]] for k in sg])
        p = pipe().fit(X[ii], y[ii]); cls = list(p.classes_)
        B[b] = p.named_steps["logisticregression"].coef_[[cls.index(c) for c in labels]]

    mean = B.mean(0)
    pv = [[2 * min(np.mean(B[:, ci, fi] > 0), np.mean(B[:, ci, fi] < 0))
           for fi in range(len(SYMPTOM28))] for ci in range(len(labels))]
    q = bh(np.array(pv).ravel()).reshape(len(labels), len(SYMPTOM28))  # 유의성(현재는 표시에 미사용)

    # ── 플롯: 클래스별 상위(top) 양의 계수 ──────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 3.7))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#DBDBDB", linewidth=0.8)

    xpos, xticks, xlabels = 0, [], []
    for ci, c in enumerate(labels):
        idx = [i for i in np.argsort(-mean[ci]) if mean[ci][i] > 0][:args.top]
        for i in idx:
            # 유의성 강조: 유의(q<0.05)하면 검은 테두리+진한 색, 아니면 흐리게
            sig = q[ci][i] < 0.05
            ax.bar(xpos, mean[ci][i], color=CLASS_COLOR[c], width=0.82, zorder=3,
                   edgecolor="black" if sig else "none",
                   linewidth=1.4 if sig else 0, alpha=1.0 if sig else 0.55)
            xticks.append(xpos); xlabels.append(SYMPTOM28[i]); xpos += 1
        xpos += 0.9  # 클래스 간 간격

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Standardized coefficient", fontsize=10)
    ax.set_ylim(bottom=0)
    ax.margins(x=0.01)

    # 상단 가로 범례 (Title Case, 네모 마커)
    # 마커 테두리: 얇은 연회색 (검정 테두리 제거). 완전히 없애려면 edgecolor="none"
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=CLASS_COLOR[c],
                             edgecolor="#C8C8C8", linewidth=0.6) for c in labels]
    ax.legend(handles, [c.title() for c in labels],
              ncol=len(labels), loc="lower center", bbox_to_anchor=(0.5, 1.0),
              frameon=False, fontsize=9.5, handlelength=1.5, handleheight=1.0,
              columnspacing=1.7, handletextpad=0.5)

    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()