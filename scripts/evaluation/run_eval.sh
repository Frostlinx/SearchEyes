#!/bin/bash
# ═══════════════════════════════════════════════════════════
# run_eval.sh — Multi-Turn Agentic Evaluation for SearchEyes
# ═══════════════════════════════════════════════════════════
#
# Usage:
#   1. Local KB eval (PKC test set, no API key needed):
#      bash run_eval.sh pkc-test
#
#   2. Web search eval (needs Serper API key):
#      export SERPER_API_KEY=your_key_here
#      bash run_eval.sh simplevqa
#
#   3. Custom model path:
#      MODEL_PATH=/path/to/model bash run_eval.sh simplevqa
#
#   4. Use already-running vLLM server:
#      VLLM_RUNNING=1 bash run_eval.sh simplevqa

set -euo pipefail

# ── Defaults ──
BENCHMARK="${1:-pkc-test}"
MODEL_PATH="${MODEL_PATH:-./Searcheyes-9b-sft}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"
MAX_TURNS="${MAX_TURNS:-10}"
VLLM_PORT="${VLLM_PORT:-8000}"
TP_SIZE="${TP_SIZE:-1}"
VLLM_RUNNING="${VLLM_RUNNING:-0}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "═══════════════════════════════════════════════"
echo " SearchEyes Multi-Turn Evaluation"
echo "═══════════════════════════════════════════════"
echo " Benchmark:     $BENCHMARK"
echo " Model:         $MODEL_PATH"
echo " Max samples:   $MAX_SAMPLES"
echo " Max turns:     $MAX_TURNS"
echo "═══════════════════════════════════════════════"

# ── Activate conda env if available ──
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate roll 2>/dev/null || true
fi

# ── Start vLLM server if needed ──
if [ "$VLLM_RUNNING" = "0" ]; then
    echo "[INFO] Starting vLLM server on port $VLLM_PORT ..."
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --port "$VLLM_PORT" \
        --tensor-parallel-size "$TP_SIZE" \
        --gpu-memory-utilization 0.9 \
        --max-model-len 16384 \
        --trust-remote-code &
    VLLM_PID=$!
    echo "[INFO] vLLM server PID: $VLLM_PID"

    # Wait for server to be ready
    echo "[INFO] Waiting for vLLM server to start ..."
    for i in $(seq 1 120); do
        if curl -s "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; then
            echo "[INFO] vLLM server ready!"
            break
        fi
        if [ "$i" -eq 120 ]; then
            echo "[ERROR] vLLM server failed to start within 240s"
            kill "$VLLM_PID" 2>/dev/null || true
            exit 1
        fi
        sleep 2
    done

    cleanup() {
        echo "[INFO] Stopping vLLM server (PID: $VLLM_PID) ..."
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    }
    trap cleanup EXIT
fi

# ── Determine search backend and data path ──
SEARCH_BACKEND="local"
EXTRA_ARGS=""

case "$BENCHMARK" in
    pkc-test)
        SEARCH_BACKEND="local"
        DATA_PATH="${DATA_PATH:-/tmp/sft_output/trajectories_correct.jsonl}"
        EXTRA_ARGS="--kb-path /tmp/pgkc_full_kb.json"
        ;;
    simplevqa)
        SEARCH_BACKEND="${SEARCH_BACKEND_OVERRIDE:-serper}"
        DATA_PATH="${DATA_PATH:-data/simplevqa/test.jsonl}"
        ;;
    infoseek)
        SEARCH_BACKEND="${SEARCH_BACKEND_OVERRIDE:-serper}"
        DATA_PATH="${DATA_PATH:-data/infoseek/test.jsonl}"
        ;;
    fvqa)
        SEARCH_BACKEND="${SEARCH_BACKEND_OVERRIDE:-serper}"
        DATA_PATH="${DATA_PATH:-data/fvqa/test.jsonl}"
        ;;
    *)
        echo "[ERROR] Unknown benchmark: $BENCHMARK"
        echo "Supported: pkc-test, simplevqa, infoseek, fvqa"
        exit 1
        ;;
esac

# Check Serper API key for web search
if [ "$SEARCH_BACKEND" = "serper" ] && [ -z "${SERPER_API_KEY:-}" ]; then
    echo "[ERROR] SERPER_API_KEY not set. Required for web search benchmarks."
    echo "  Get one at https://serper.dev (free tier: 2500 queries)"
    echo "  Then: export SERPER_API_KEY=your_key_here"
    exit 1
fi

# ── Run evaluation ──
echo ""
echo "[INFO] Running evaluation ..."
python eval_multiturn.py \
    --api-base "http://localhost:${VLLM_PORT}/v1" \
    --model-name "$MODEL_PATH" \
    --benchmark "$BENCHMARK" \
    --data-path "$DATA_PATH" \
    --search-backend "$SEARCH_BACKEND" \
    --max-samples "$MAX_SAMPLES" \
    --max-turns "$MAX_TURNS" \
    $EXTRA_ARGS

echo ""
echo "[INFO] Evaluation complete!"
