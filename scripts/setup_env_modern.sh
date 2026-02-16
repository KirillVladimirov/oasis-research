#!/usr/bin/env bash
# scripts/setup_env_modern.sh
#
# Создаёт .venv-modern (Python 3.12) и ставит минимальные зависимости для BERTopic.
# По умолчанию: CPU torch. Для CUDA cu118: --cuda
#
# Фиксы:
# - uv pip uninstall НЕ поддерживает -y -> убрано
# - Решение "No space left on device" при распаковке nvidia-* wheel'ов:
#   перенос TMPDIR и UV_CACHE_DIR на диск с местом
# - pip может отсутствовать в venv -> добавлен bootstrap pip (ensurepip -> fallback uv)
# - Кэш uv включён, чтобы пакеты не скачивались заново

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

USE_CUDA=0
PYTHON_VER="3.12"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda)
      USE_CUDA=1
      shift
      ;;
    --python)
      PYTHON_VER="${2:-3.12}"
      shift 2
      ;;
    -h|--help)
      cat <<'HELP'
Usage:
  ./scripts/setup_env_modern.sh [--cuda] [--python 3.12]

Options:
  --cuda         Install PyTorch with CUDA 11.8 wheels (cu118). Needs disk space for extraction.
  --python X.Y   Python version for uv venv (default: 3.12)
HELP
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Run with --help"
      exit 1
      ;;
  esac
done

echo "Creating modern environment (.venv-modern) in: $ROOT_DIR"
echo "Python: $PYTHON_VER"
echo "CUDA install: $USE_CUDA"

# -------------------------
# Temp + cache (critical)
# -------------------------
TMP_BASE="${TMPDIR:-$HOME/_tmp}"
UV_CACHE_BASE="${UV_CACHE_DIR:-$HOME/_uv_cache}"

mkdir -p "$TMP_BASE" "$UV_CACHE_BASE"

export TMPDIR="$TMP_BASE"
export UV_CACHE_DIR="$UV_CACHE_BASE"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-10}"

# Важно: не выключать кэш
unset UV_NO_CACHE || true

echo "TMPDIR=$TMPDIR"
echo "UV_CACHE_DIR=$UV_CACHE_DIR"
echo "UV_HTTP_TIMEOUT=$UV_HTTP_TIMEOUT"
echo "UV_HTTP_RETRIES=$UV_HTTP_RETRIES"

if [[ "$USE_CUDA" -eq 1 ]]; then
  AVAIL_KB="$(df -Pk "$TMPDIR" | awk 'NR==2{print $4}')"
  AVAIL_GB="$((AVAIL_KB / 1024 / 1024))"
  if [[ "$AVAIL_GB" -lt 15 ]]; then
    echo "WARNING: Free space in TMPDIR is only ${AVAIL_GB}GB."
    echo "CUDA wheels may fail to extract with 'No space left on device'."
    echo "Set TMPDIR to a disk with >= 15-20GB free space and rerun."
  fi
fi

# -------------------------
# Recreate venv
# -------------------------
if [[ -d ".venv-modern" ]]; then
  echo "Removing existing .venv-modern..."
  rm -rf .venv-modern
fi

echo "Creating venv via uv..."
uv venv --python "$PYTHON_VER" .venv-modern

# Activate venv
# shellcheck disable=SC1091
source .venv-modern/bin/activate

# -------------------------
# Bootstrap pip (uv venv can be created without pip)
# -------------------------
if ! python -c "import pip" >/dev/null 2>&1; then
  echo "pip is missing in venv; trying ensurepip..."
  python -m ensurepip --upgrade >/dev/null 2>&1 || true
fi

if ! python -c "import pip" >/dev/null 2>&1; then
  echo "ensurepip did not provide pip; installing pip via uv..."
  uv pip install pip setuptools wheel
else
  python -m pip install -U pip setuptools wheel >/dev/null
fi

# -------------------------
# Base deps
# -------------------------
echo "Installing base dependencies..."
uv pip install \
  loguru \
  tqdm \
  python-dotenv \
  numpy \
  scikit-learn \
  "umap-learn==0.5.9.post2" \
  "hdbscan==0.8.41"

# -------------------------
# PyTorch (minimal)
# -------------------------
echo "Removing existing torch stack (if any)..."
uv pip uninstall torch torchvision torchaudio torchtext || true

if [[ "$USE_CUDA" -eq 1 ]]; then
  echo "Installing PyTorch (CUDA 11.8 / cu118)..."
  uv pip install "torch==2.7.1" --index-url https://download.pytorch.org/whl/cu118
else
  echo "Installing PyTorch (CPU-only)..."
  uv pip install "torch==2.7.1" --index-url https://download.pytorch.org/whl/cpu
fi

# -------------------------
# NLP stack
# -------------------------
echo "Installing transformers + sentence-transformers + bertopic..."
uv pip install \
  "transformers==4.57.3" \
  "sentence-transformers==5.2.0" \
  "bertopic==0.17.4" \
  safetensors

# -------------------------
# Sanity check
# -------------------------
echo ""
echo "Sanity check:"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    try:
        print("arch list:", torch.cuda.get_arch_list())
    except Exception as e:
        print("arch list error:", e)
PY

deactivate

echo ""
echo "Done."
echo "Activate: source .venv-modern/bin/activate"
echo "Run BERTopic: python3 scripts/topic_modeling_bertopic/02_train_bertopic.py"
