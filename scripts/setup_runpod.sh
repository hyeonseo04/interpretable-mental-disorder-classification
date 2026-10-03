#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_runpod.sh — RunPod 환경 구성 (uv.lock 기반 버전 고정) + 모델 다운로드 + 검증
#
#   cd /workspace/interpretable-mental-disorder-classification
#   bash scripts/setup_runpod.sh
#
# 환경변수 (선택)
#   WORK=/workspace              네트워크 볼륨 루트 (모델·캐시 저장 위치)
#   SKIP_MODELS=1                모델 다운로드 생략
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
WORK=${WORK:-/workspace}
export HF_HOME=${HF_HOME:-$WORK/hf}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$WORK/.uv-cache}
export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-$WORK/.uv-python}
PY=3.12.10

say() { printf "\n\033[1m== %s\033[0m\n" "$*"; }

say "1. GPU / 드라이버"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
NGPU=$(nvidia-smi -L | wc -l)
MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$DRV" -lt 570 ]; then
  echo "✗ 드라이버 $DRV < 570: PyTorch 2.10.0(CUDA 12.8 빌드)이 동작하지 않을 수 있습니다."
  echo "  RunPod 에서 Pod 생성 시 'CUDA Version' 필터를 12.8 이상으로 고르세요."; exit 1
fi
TOTAL=$((NGPU * MEM))
if [ "$NGPU" -eq 1 ] && [ "$MEM" -lt 100000 ]; then
  echo "⚠ GPU 1장(${MEM}MiB): 14B 는 가능하지만 32B(bf16 가중치 ~65.5GB)는 32,768 컨텍스트용 KV 캐시"
  echo "  (~8.6GB)를 확보하지 못할 가능성이 큽니다 → 32B 는 GPU 2장(tp=2) 또는 H200 1장을 쓰세요."
fi

say "2. uv / Python $PY"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version
uv python install "$PY"

say "3. 패키지 설치"
if [ -f uv.lock ] && [ -f pyproject.toml ]; then
  echo "uv.lock 사용 (frozen — lock 과 다른 버전은 설치되지 않음)"
  uv sync --frozen --no-install-project --python "$PY"
  # 그림용 matplotlib 이 lock 에 없으면 분석용으로만 추가
  .venv/bin/python -c "import matplotlib" 2>/dev/null || uv pip install --python .venv/bin/python matplotlib
else
  echo "⚠ uv.lock 이 없습니다. requirements-pinned.txt 로 직접 버전을 고정합니다"
  echo "  (직접 의존성만 고정 — 전이 의존성까지 같게 하려면 원본 uv.lock 을 복사하세요)."
  uv venv --python "$PY" .venv
  uv pip install --python .venv/bin/python -r requirements-pinned.txt
fi

say "4. 버전 검증 (논문 3.5절)"
PYTHONPATH=src .venv/bin/python -m imdc.env

say "5. 모델 다운로드 (HF_HOME=$HF_HOME)"
: > .runpod_env
echo "export HF_HOME=$HF_HOME" >> .runpod_env
echo "export NGPU=$NGPU" >> .runpod_env
echo "export GPU_MEM_MIB=$MEM" >> .runpod_env
if [ "${SKIP_MODELS:-0}" != "1" ]; then
  for KEY in 14b 32b; do
    if [ $KEY = 14b ]; then ID=Qwen/Qwen2.5-14B-Instruct; WANT=cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8;
    else ID=Qwen/Qwen2.5-32B-Instruct; WANT=5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd; fi
    REV=$WANT    # 실험에 쓴 커밋 해시로 고정
    P=$(.venv/bin/hf download "$ID" --revision "$REV" --quiet 2>/dev/null | tail -1 || true)
    if [ -z "$P" ] || [ ! -d "$P" ]; then
      P=$(.venv/bin/python -c "from huggingface_hub import snapshot_download as s; print(s('$ID', revision='$REV'))")
    fi
    SNAP=$(basename "$P")
    if [ "$SNAP" = "$WANT" ]; then echo "✓ $ID snapshot $SNAP (고정 리비전과 일치)";
    else echo "✗ $ID snapshot $SNAP ≠ 고정 리비전 $WANT"; exit 1; fi
    echo "export MODEL_${KEY^^}=$P" >> .runpod_env
  done
fi
cat .runpod_env

say "완료. 다음 단계"
cat <<'EOF'
  1) 데이터 업로드:  data/raw/Training/02.라벨링데이터 , data/raw/Validation/02.라벨링데이터 (이 저장소 안)
  2) tmux new -s imdc
  3) bash scripts/run_all.sh          # 매니페스트 생성 → 토큰 점검 → 8개 구성 (중단 시 같은 명령으로 재개)
EOF
