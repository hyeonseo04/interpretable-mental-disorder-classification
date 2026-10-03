# -*- coding: utf-8 -*-
"""
verify.py — 재실행 결과가 저장소의 expected/ (논문 수치)와 같은지 점검.

  python -m imdc.verify                       # paper_out/ · results/ · data/processed/ 를 expected/ 와 비교
  python -m imdc.verify --tol 0.005           # 점 추정치 허용 오차 (기본 0: 완전 일치)

비교 항목
  1) data/processed/manifest_meta.json : 세션·내담자·손상 수, 매니페스트 해시 (같은 데이터인지)
  2) results/*.done                    : 8개 구성의 기록 수·파싱 상태
  3) paper_out/table3·table4           : 모든 점 추정치 (부트스트랩 CI 열은 제외)
주의: greedy 디코딩이라도 GPU 종류·tensor parallel 크기가 다르면 LLM 출력이 미세하게 달라질 수 있다.
      expected/ 는 RTX A6000 × 2 (tp=2) 에서 만든 값이다.
"""
import argparse
import csv
import glob
import json
import os
import sys

CI_COLS = {"CI_low", "CI_high"}


def read_csv(p):
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.reader(f))


def cmp_table(a_path, b_path, tol):
    a, b = read_csv(a_path), read_csv(b_path)
    if a[0] != b[0] or len(a) != len(b):
        return [f"구조가 다름 (행 {len(a)} vs {len(b)})"]
    skip = {j for j, h in enumerate(a[0]) if h in CI_COLS}
    diffs = []
    for i, (ra, rb) in enumerate(zip(a[1:], b[1:]), 1):
        for j, (x, y) in enumerate(zip(ra, rb)):
            if j in skip or x == y:
                continue
            try:
                if abs(float(x) - float(y)) <= tol:
                    continue
            except ValueError:
                pass
            diffs.append(f"{ra[0]}/{ra[1] if len(ra) > 1 else ''} [{a[0][j]}]: {x} vs 기대 {y}")
    return diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expected", default="expected")
    ap.add_argument("--paper-dir", default="paper_out")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--tol", type=float, default=0.0)
    args = ap.parse_args()
    fails = 0

    def report(name, problems):
        nonlocal fails
        if problems:
            fails += 1
            print(f"✗ {name}")
            for p in problems[:20]:
                print(f"    {p}")
        else:
            print(f"✓ {name}")

    # 1) 데이터
    e = json.load(open(os.path.join(args.expected, "runs", "manifest_meta.json")))
    mp = os.path.join(args.processed_dir, "manifest_meta.json")
    if os.path.exists(mp):
        g = json.load(open(mp))
        report("데이터(매니페스트)", [f"{k}: {g.get(k)} vs 기대 {e.get(k)}"
                                  for k in ("n_sessions", "n_persons", "n_corrupted", "manifest_sha1")
                                  if g.get(k) != e.get(k)])
    else:
        report("데이터(매니페스트)", [f"{mp} 없음 — python -m imdc.data 먼저 실행"])

    # 2) 추론 결과
    probs = []
    for ed in sorted(glob.glob(os.path.join(args.expected, "runs", "*.done"))):
        name = os.path.basename(ed)
        gd = os.path.join(args.results_dir, name)
        if not os.path.exists(gd):
            probs.append(f"{name} 없음"); continue
        x, y = json.load(open(gd)), json.load(open(ed))
        for k in ("n", "parse_status", "finish_length"):
            if x.get(k) != y.get(k):
                probs.append(f"{name} {k}: {x.get(k)} vs 기대 {y.get(k)}")
    report("추론 결과 8개 구성", probs)

    # 3) 표
    for t in ("table3_performance.csv", "table4_faithfulness.csv"):
        gp = os.path.join(args.paper_dir, t)
        if not os.path.exists(gp):
            report(t, [f"{gp} 없음 — bash scripts/make_paper.sh 먼저 실행"]); continue
        report(t, cmp_table(gp, os.path.join(args.expected, "paper", t), args.tol))

    print("\n결과: " + ("모두 일치" if not fails else f"{fails}개 항목 불일치"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
