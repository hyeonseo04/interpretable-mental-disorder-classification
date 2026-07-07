#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
symptom_factor 28개 EDA
- 입력: data_loader.py가 만든 sessions.jsonl
- ① 클래스별 평균 강도 / ≥2 비율 (factor × class) → 변별 구조 확인
- ② 사람 단위 gold 집계 (사람별 세션 max) → "각 사람 gold 뽑는 법"
- ③ 히트맵 PNG 저장
사용: python3 eda_symptom.py sessions.jsonl
"""
import sys, json, statistics
from collections import defaultdict

INP = sys.argv[1] if len(sys.argv) > 1 else "sessions.jsonl"

# 공식 symptom_factor 3-1~3-28 (self_control 제외)
SYMPTOM28 = [
 "depressive_mood","worthlessness","guilt","impaired_cognition","suicidal","anhedonia",
 "psychomotor_changes","weight_appetite","sleep_disturbance","fatigue",          # 우울(3-1~10)
 "anxiety_mood","derealization","perceived_loss_of_control","anxiety_control",
 "concentration","avoidance","physical_symptoms","irritability",                  # 불안(3-11~18)
 "loss_of_control","craving","lying","tolerance","withdrawal","salience",
 "resource_investment","daily_functioning","social_problems","negative_consequences"]  # 중독(3-19~28)
assert len(SYMPTOM28) == 28
CLASSES = ["DEPRESSION","ANXIETY","ADDICTION","NORMAL"]

rows = [json.loads(l) for l in open(INP, encoding="utf-8")]
rows = [r for r in rows if r.get("class") in CLASSES]
print(f"세션 {len(rows)}개\n")

# 세션 gold raw(0~3) 수집
vals = {c: {f: [] for f in SYMPTOM28} for c in CLASSES}
for r in rows:
    gf = r.get("gold_factors_raw", {})
    for f in SYMPTOM28:
        vals[r["class"]][f].append(gf.get(f, 0))

# ① 클래스별 평균 강도
print("=== ① 클래스별 평균 강도 (0~3) · factor × class ===")
print("factor".ljust(26) + "".join(c[:4].rjust(8) for c in CLASSES) + "   top")
for f in SYMPTOM28:
    m = {c: statistics.mean(vals[c][f]) for c in CLASSES}
    top = max(m, key=m.get)
    print(f.ljust(26) + "".join(f"{m[c]:.2f}".rjust(8) for c in CLASSES) + "   " + top)

# ② 클래스별 ≥2 비율
print("\n=== ② 클래스별 ≥2 비율 · factor × class ===")
frac = {c: {} for c in CLASSES}
for f in SYMPTOM28:
    for c in CLASSES:
        frac[c][f] = sum(1 for v in vals[c][f] if v >= 2) / max(len(vals[c][f]), 1)
    top = max(CLASSES, key=lambda c: frac[c][f])
    print(f.ljust(26) + "".join(f"{frac[c][f]*100:5.0f}%".rjust(8) for c in CLASSES) + "   " + top)

# ③ 사람 단위 gold 집계 (사람별 세션 max)
print("\n=== ③ 사람 단위 gold 집계 예시 (사람별 세션 max, ≥2만) ===")
person = defaultdict(lambda: defaultdict(int)); pcls = {}
for r in rows:
    pid = r["person_id"]; pcls[pid] = r["class"]
    gf = r.get("gold_factors_raw", {})
    for f in SYMPTOM28:
        person[pid][f] = max(person[pid][f], gf.get(f, 0))
shown = set()
for pid, fac in person.items():
    c = pcls[pid]
    if c in shown: continue
    shown.add(c)
    nz = {f: v for f, v in fac.items() if v >= 2}
    print(f"[{c}] {pid}: {dict(sorted(nz.items(), key=lambda x:-x[1]))}")
print(f"\n총 사람 수: {len(person)} (클래스별: "
      + ", ".join(f'{c} {sum(1 for p in pcls.values() if p==c)}' for c in CLASSES) + ")")

# ④ 히트맵 저장
try:
    import numpy as np, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    M = np.array([[frac[c][f] for c in CLASSES] for f in SYMPTOM28])  # 28 x 4
    fig, ax = plt.subplots(figsize=(6, 11))
    im = ax.imshow(M, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES, rotation=30, ha="right")
    ax.set_yticks(range(len(SYMPTOM28))); ax.set_yticklabels(SYMPTOM28, fontsize=8)
    ax.set_title("symptom_factor ≥2 비율 (factor × class)")
    for i in range(28):
        for j in range(4):
            ax.text(j, i, f"{M[i,j]*100:.0f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout(); plt.savefig("eda_symptom_heatmap.png", dpi=130)
    print("\n[저장] eda_symptom_heatmap.png")
except Exception as e:
    print(f"\n(히트맵 생략: {e})")

if __name__ == "__main__":
    pass