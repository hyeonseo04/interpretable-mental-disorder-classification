#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_extraction.py — LLM 추출 요인 vs 전문가 골드 충실성 검증 (논문 4.3)

  session_gold.jsonl 과 session_llm_{tag}.jsonl 을 session_id 기준으로 정렬 후,
  factor별로 다음을 계산한다:
    - Spearman ρ            : 순위 일치도
    - Quadratic Weighted Kappa
    - recall                : 골드≥2 세션 중 LLM≥2 비율
    - 골드 양성(≥2) 세션 수 : 표본 충분성

  골드 양성이 MIN_POS 미만인 factor 는 '데이터 부족'으로 표시하고 평균 집계에서 제외한다.
  (이 목록이 4.3의 '검증 제외 요인'과 일치해야 함 — 논문·gold_aggregate.py 와 기준 통일)

데이터 규약: session_id 정규화(_raw_→_check_, person 대문자) — gold/llm 모두 동일 적용.

사용:
  uv run --with scipy --with scikit-learn validate_extraction.py \
      --gold ../results/session_gold.jsonl --llm ../results/session_llm_32b.jsonl
"""

import argparse
import json
import re

from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

SYMPTOM28 = [
    "depressive_mood", "worthlessness", "guilt", "impaired_cognition", "suicidal", "anhedonia",
    "psychomotor_changes", "weight_appetite", "sleep_disturbance", "fatigue",
    "anxiety_mood", "derealization", "perceived_loss_of_control", "anxiety_control",
    "concentration", "avoidance", "physical_symptoms", "irritability",
    "loss_of_control", "craving", "lying", "tolerance", "withdrawal", "salience",
    "resource_investment", "daily_functioning", "social_problems", "negative_consequences",
]


def normalize_sid(name: str) -> str:
    sid = str(name).strip().replace("_raw_", "_check_")
    return re.sub(r"(_check_)([a-z])", lambda m: m.group(1) + m.group(2).upper(), sid)


def read(path, key):
    d = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        d[normalize_sid(r["session_id"])] = r[key]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="result/session_gold.jsonl")
    ap.add_argument("--llm", default="result/session_llm_14b.jsonl")
    ap.add_argument("--min-pos", type=int, default=30,
                    help="골드 양성(≥2) 세션 수가 이 미만이면 검증에서 제외")
    args = ap.parse_args()

    gold = read(args.gold, "gold")
    llm = read(args.llm, "llm")
    common = sorted(set(gold) & set(llm))
    print(f"gold {len(gold)} · llm {len(llm)} · 공통 {len(common)} 세션\n")
    if not common:
        print("⚠️ 공통 session_id 0개 — gold/llm 의 session_id 규약을 확인하세요.")
        return

    print(f"{'factor':26}{'n':>5}{'골드≥2':>7}{'Spearman':>10}{'QWK':>8}{'recall':>8}  비고")
    rows = []
    for f in SYMPTOM28:
        g = [gold[s].get(f, 0) for s in common]
        l = [llm[s].get(f, 0) for s in common]
        pos2 = sum(1 for x in g if x >= 2)
        rho = (spearmanr(g, l).correlation
               if len(set(g)) > 1 and len(set(l)) > 1 else float("nan"))
        try:
            qwk = cohen_kappa_score(g, l, weights="quadratic", labels=[0, 1, 2, 3])
        except Exception:
            qwk = float("nan")
        pair = [(gg, ll) for gg, ll in zip(g, l) if gg >= 2]
        rec = (sum(1 for gg, ll in pair if ll >= 2) / len(pair)) if pair else float("nan")
        flag = "" if pos2 >= args.min_pos else "데이터 부족"
        rows.append((f, pos2, rho, qwk, rec, flag))
        print(f"{f:26}{len(common):>5}{pos2:>7}{rho:>10.3f}{qwk:>8.3f}{rec:>8.2f}  {flag}")

    ok = [r for r in rows if r[1] >= args.min_pos]

    def avg(i):
        vs = [r[i] for r in ok if r[i] == r[i]]   # nan 제외
        return sum(vs) / len(vs) if vs else float("nan")

    print(f"\n[검증 가능 factor {len(ok)}개 평균]  "
          f"Spearman {avg(2):.3f} · QWK {avg(3):.3f} · recall {avg(4):.2f}")
    dropped = [r[0] for r in rows if r[1] < args.min_pos]
    print(f"[데이터 부족 {len(dropped)}개]: " + ", ".join(dropped))


if __name__ == "__main__":
    main()