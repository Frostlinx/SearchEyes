#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON="${PYTHON_BIN}"
elif [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON="${ROOT_DIR}/.venv/bin/python"
else
  PYTHON="python3"
fi

MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/QWEN/Qwen3-VL-Embedding-2B}"
EMBEDDING_PORT="${EMBEDDING_PORT:-8766}"
EMBEDDING_URL="${EMBEDDING_URL:-http://localhost:${EMBEDDING_PORT}}"
DTYPE="${DTYPE:-auto}"
LOG_DIR="${ROOT_DIR}/output/logs"
LOG_FILE="${LOG_DIR}/embedding_server_${EMBEDDING_PORT}.log"
PID_FILE="${LOG_DIR}/embedding_server_${EMBEDDING_PORT}.pid"
RAG_DB_PATH="${RAG_DB_PATH:-data/wit_kb_v2/chroma_db}"

mkdir -p "${LOG_DIR}"

if [[ "${RAG_DB_PATH}" = /* ]]; then
  RAG_DB_ABS="${RAG_DB_PATH}"
else
  RAG_DB_ABS="${ROOT_DIR}/${RAG_DB_PATH}"
fi

if [[ ! -d "${RAG_DB_ABS}" ]]; then
  for candidate in \
    "${ROOT_DIR}/data/wit_subset_hf/chroma_db" \
    "${ROOT_DIR}/data/wit_subset/chroma_db"
  do
    if [[ -d "${candidate}" ]]; then
      RAG_DB_ABS="${candidate}"
      break
    fi
  done
fi

if [[ -f "${PID_FILE}" ]]; then
  EXISTING_PID="$(cat "${PID_FILE}")"
  if kill -0 "${EXISTING_PID}" >/dev/null 2>&1; then
    echo "[start-env] embedding server already running: pid=${EXISTING_PID}"
    echo "[start-env] log: ${LOG_FILE}"
    echo "[start-env] smoke: ${PYTHON} ${ROOT_DIR}/scripts/smoke_test_rag_v2.py --embedding-url ${EMBEDDING_URL} --db-path ${RAG_DB_ABS}"
    exit 0
  fi
fi

if command -v lsof >/dev/null 2>&1; then
  if lsof -iTCP:"${EMBEDDING_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "[start-env] port ${EMBEDDING_PORT} is already in use"
    echo "[start-env] smoke: ${PYTHON} ${ROOT_DIR}/scripts/smoke_test_rag_v2.py --embedding-url ${EMBEDDING_URL} --db-path ${RAG_DB_ABS}"
    exit 1
  fi
fi

nohup "${PYTHON}" "${ROOT_DIR}/searcheyes/embedding_server.py" \
  --model-path "${MODEL_PATH}" \
  --dtype "${DTYPE}" \
  --port "${EMBEDDING_PORT}" \
  >"${LOG_FILE}" 2>&1 &

SERVER_PID=$!
echo "${SERVER_PID}" > "${PID_FILE}"

echo "[start-env] embedding server started"
echo "[start-env] pid: ${SERVER_PID}"
echo "[start-env] log: ${LOG_FILE}"
echo "[start-env] health: curl ${EMBEDDING_URL}/health"
echo "[start-env] smoke: ${PYTHON} ${ROOT_DIR}/scripts/smoke_test_rag_v2.py --embedding-url ${EMBEDDING_URL} --db-path ${RAG_DB_ABS}"
