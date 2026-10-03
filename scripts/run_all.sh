#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_all.sh — 8개 구성(4 프롬프트 × 14B/32B) 재추출. 중단되면 **같은 명령을 다시 실행**하면 이어서 돈다.
#
#   tmux new -s imdc
#   bash scripts/run_all.sh
#
# 순서: [0] 매니페스트(전사 절단 없음) → [1] 토큰 길이 점검(dry-run) →
#       [2] 14B: factor28 → zeroshot → fewshot → direct4 → [3] 32B: 같은 순서
#   - 모델당 1회 로드로 4개 방법을 연속 수행
#   - 각 방법은 청크 단위로 저장·재개, 완료 시 results/*.done 생성 → 다음 실행에서 건너뜀
#
# 환경변수 (선택)
#   DATA_TRAIN / DATA_VALID   원본 라벨링 데이터 경로
#   TP_14B / TP_32B           tensor parallel 크기 (기본: GPU 2장 이상이면 2, 1장이면 1)
#   GPU_UTIL=0.92             gpu_memory_utilization
#   MAX_NUM_SEQS=             OOM 시 4 등으로 제한
#   MODELS="14b 32b"          일부 모델만
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .runpod_env ] && source .runpod_env
export PYTHONPATH=src
PY=.venv/bin/python
DATA_TRAIN=${DATA_TRAIN:-data/raw/Training/02.라벨링데이터}
DATA_VALID=${DATA_VALID:-data/raw/Validation/02.라벨링데이터}
NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}
DEF_TP=$([ "$NGPU" -ge 2 ] && echo 2 || echo 1)
TP_14B=${TP_14B:-$DEF_TP}
TP_32B=${TP_32B:-$DEF_TP}
GPU_UTIL=${GPU_UTIL:-0.92}
MODELS=${MODELS:-"14b 32b"}
METHODS="factor28 zeroshot fewshot direct4"
mkdir -p logs results
LOG=logs/run_all_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
echo "[$(date '+%F %T')] run_all 시작 · GPU ${NGPU}장 · TP 14B=$TP_14B 32B=$TP_32B · 로그 $LOG"

[ "${SKIP_ENV_CHECK:-0}" = 1 ] || $PY -m imdc.env   # 버전이 논문과 다르면 여기서 중단 (SKIP_ENV_CHECK 는 테스트 전용)

if [ ! -f data/processed/manifest.jsonl ]; then
  echo "── [0] 매니페스트 생성"
  $PY -m imdc.data "$DATA_TRAIN" "$DATA_VALID" --out-dir data/processed
fi

for KEY in $MODELS; do
  VAR="MODEL_${KEY^^}"; MP=${!VAR:-}
  [ -z "$MP" ] && { echo "✗ $VAR 가 없습니다 (setup_runpod.sh 를 먼저 실행)"; exit 1; }
  TPVAR="TP_${KEY^^}"; TP=${!TPVAR}
  ALL_DONE=1
  for M in $METHODS; do
    case $M in factor28) S=session_llm;; zeroshot) S=session_direct_zeroshot;;
               fewshot) S=session_direct_fewshot;; direct4) S=session_direct4;; esac
    [ -f "results/${S}_${KEY}_full.done" ] || ALL_DONE=0
  done
  if [ $ALL_DONE = 1 ]; then echo "── $KEY: 4개 방법 모두 완료 — 건너뜀"; continue; fi

  echo "── [1] $KEY 토큰 길이 점검 (모델 로드 없음)"
  $PY -m imdc.infer --model-key "$KEY" --model-path "$MP" --methods $METHODS --dry-run

  echo "── [2] $KEY 추론 (tp=$TP)"
  $PY -m imdc.infer --model-key "$KEY" --model-path "$MP" --methods $METHODS \
      --tp "$TP" --gpu-mem "$GPU_UTIL" ${MAX_NUM_SEQS:+--max-num-seqs $MAX_NUM_SEQS}
done

echo "[$(date '+%F %T')] 추론 종료. 완료 표시:"
ls -1 results/*_full.done
N=$(ls results/*_full.done 2>/dev/null | wc -l)
if [ "$N" -eq 8 ]; then
  echo "✓ 8개 구성 완료 → bash scripts/make_paper.sh"
else
  echo "✗ ${N}/8 완료 — 같은 명령으로 다시 실행하면 이어서 진행합니다."; exit 1
fi
