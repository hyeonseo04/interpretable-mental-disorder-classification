#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# make_paper.sh — 결과 → Table 3·4, Fig. 2·3 + A-1 / A-2 / C-(1) 재분석 (GPU 불필요)
#
#   bash scripts/make_paper.sh                 # 결과 (*_full.jsonl) → paper_out/, analysis_out/
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=${PY:-.venv/bin/python}
SUFFIX=${1-_full}
TAG=${2-}
RES=${RESULTS_DIR:-results}

$PY -m imdc.paper       --results-dir "$RES" --suffix "$SUFFIX" --out-dir "paper_out$TAG"
$PY -m imdc.analysis_c1 --results-dir "$RES" --suffix "$SUFFIX" --out-dir "analysis_out$TAG/c1"
$PY -m imdc.analysis_a2 --results-dir "$RES" --suffix "$SUFFIX" --out-dir "analysis_out$TAG/a2"
$PY -m imdc.analysis_a1 --results-dir "$RES" --suffix "$SUFFIX" --out-dir "analysis_out$TAG/a1"
echo "✓ paper_out$TAG/ · analysis_out$TAG/{a1,a2,c1}/"
