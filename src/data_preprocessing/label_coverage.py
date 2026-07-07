#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
골드 라벨링 충실도(coverage) 프로파일링
- 라벨이 필수(N)가 아니라 '얼마나 라벨링됐나'를 발화/세션/사람/factor 단위로 측정
- 원본 json 읽음 (json-repair 폴백)
사용: python3 label_coverage.py /home/hslee/Depression/data/Training/02.라벨링데이터
"""
import sys, os, json, glob
from collections import defaultdict, Counter

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/hslee/Depression/data/Training/02.라벨링데이터"

META = {"start_point","end_point","character_count","cps","paragraph_speaker","paragraph_text","index"}
CLIENT = "내담자"
CLASSES = ["DEPRESSION","ANXIETY","ADDICTION","NORMAL"]
SYMPTOM28 = [
 "depressive_mood","worthlessness","guilt","impaired_cognition","suicidal","anhedonia",
 "psychomotor_changes","weight_appetite","sleep_disturbance","fatigue",
 "anxiety_mood","derealization","perceived_loss_of_control","anxiety_control",
 "concentration","avoidance","physical_symptoms","irritability",
 "loss_of_control","craving","lying","tolerance","withdrawal","salience",
 "resource_investment","daily_functioning","social_problems","negative_consequences"]
SYM = set(SYMPTOM28)

def load(f):
    try:
        return json.load(open(f, encoding="utf-8"))
    except Exception:
        try:
            from json_repair import repair_json
            return json.loads(repair_json(open(f, encoding="utf-8").read()))
        except Exception:
            return None

# 카운터
client_utts = Counter(); tagged_utts = Counter()         # 발화 단위
val_dist = Counter(); per_utt_tagcount = []              # 라벨 강도/발화당 태깅수
sess_total = Counter(); sess_empty = Counter()           # 세션 단위
present = defaultdict(Counter); tag1 = defaultdict(Counter); tag2 = defaultdict(Counter)  # factor 단위
person_factors = defaultdict(set); person_cls = {}       # 사람 단위

files = glob.glob(os.path.join(DATA_DIR, "**", "*.json"), recursive=True)
for f in files:
    d = load(f)
    if not d or d.get("class") not in CLASSES:
        continue
    c = d["class"]; sess_total[c] += 1
    pid = d.get("id", os.path.basename(f)); person_cls[pid] = c
    cols_here = set(); sess_max = defaultdict(int); has_tag = False
    for p in d.get("paragraph", []):
        if not isinstance(p, dict) or p.get("paragraph_speaker") != CLIENT:
            continue
        client_utts[c] += 1; utt_tags = 0
        for k, v in p.items():
            if k in SYM and isinstance(v, (int, float)):
                cols_here.add(k)
                if v > 0:
                    utt_tags += 1; val_dist[int(v)] += 1
                    sess_max[k] = max(sess_max[k], int(v))
                    has_tag = True; person_factors[pid].add(k)
        if utt_tags:
            tagged_utts[c] += 1; per_utt_tagcount.append(utt_tags)
    if not has_tag:
        sess_empty[c] += 1
    for k in cols_here:
        present[c][k] += 1
        if sess_max[k] >= 1: tag1[c][k] += 1
        if sess_max[k] >= 2: tag2[c][k] += 1

# ── 출력 ──
print("="*60)
print("A) 전체 규모 / 발화 단위 라벨 커버리지")
print("="*60)
tot_utt = sum(client_utts.values()); tot_tag = sum(tagged_utts.values())
print(f"세션 {sum(sess_total.values())} · 내담자 발화 {tot_utt}")
print(f"라벨 달린 발화(≥1 factor) {tot_tag} = {tot_tag/max(tot_utt,1)*100:.1f}%  "
      f"(나머지 {100-tot_tag/max(tot_utt,1)*100:.1f}%는 라벨 없음)")
print("\n[클래스별]")
for c in CLASSES:
    if client_utts[c]:
        print(f"  {c}: 발화 {client_utts[c]}, 라벨 발화 {tagged_utts[c]} "
              f"({tagged_utts[c]/client_utts[c]*100:.1f}%)")

print("\n" + "="*60)
print("B) 라벨 강도 분포 / 발화당 태깅 factor 수")
print("="*60)
print(f"강도 분포: 1점 {val_dist[1]}, 2점 {val_dist[2]}, 3점 {val_dist[3]}")
if per_utt_tagcount:
    print(f"라벨 달린 발화의 발화당 factor 수: 평균 {sum(per_utt_tagcount)/len(per_utt_tagcount):.2f}, "
          f"최대 {max(per_utt_tagcount)}")

print("\n" + "="*60)
print("C) 세션 단위 — 라벨이 하나도 없는 세션")
print("="*60)
for c in CLASSES:
    if sess_total[c]:
        print(f"  {c}: 세션 {sess_total[c]}, 빈 세션 {sess_empty[c]} "
              f"({sess_empty[c]/sess_total[c]*100:.0f}%)")

print("\n" + "="*60)
print("D) 사람 단위 — 태깅된 factor 종류 수 / 빈 사람")
print("="*60)
pc = Counter(person_cls.values())
emptyp = Counter(c for pid,c in person_cls.items() if len(person_factors[pid])==0)
for c in CLASSES:
    if pc[c]:
        kinds = [len(person_factors[pid]) for pid,cc in person_cls.items() if cc==c]
        print(f"  {c}: 사람 {pc[c]}, 사람당 태깅 factor종류 평균 {sum(kinds)/len(kinds):.1f}, "
              f"빈 사람 {emptyp[c]} ({emptyp[c]/pc[c]*100:.0f}%)")

print("\n" + "="*60)
print("D) factor별 라벨링 정도 (컬럼 존재 세션 기준)")
print("="*60)
print("factor".ljust(26) + "present  ≥1%   ≥2%   존재클래스")
for f in SYMPTOM28:
    pr = sum(present[c][f] for c in CLASSES)
    if not pr: 
        print(f.ljust(26) + "   0    (컬럼 없음)")
        continue
    t1 = sum(tag1[c][f] for c in CLASSES); t2 = sum(tag2[c][f] for c in CLASSES)
    cls = ",".join(c[:4] for c in CLASSES if present[c][f] > 0)
    print(f.ljust(26) + f"{pr:6d} {t1/pr*100:5.0f}% {t2/pr*100:5.0f}%   {cls}")

if __name__ == "__main__":
    pass