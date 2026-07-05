#!/bin/bash
# Run VisSearch Bench evaluation for all available API models
# Each model runs in background as a separate process

cd .
OUTDIR=outputs/eval_multiturn
mkdir -p $OUTDIR

API_BASE="https://routify.alibaba-inc.com/protocol/openai/v1"
API_KEY="sk-9eba1adb38fa4cb1af5dca05f58f8472"
KB_PATH="/tmp/pgkc_full_kb.json"
DATA_PATH="data/vissearch_bench.jsonl"
MAX_SAMPLES=1000

# Models to evaluate
MODELS=(
  "gpt-4o"
  "gpt-5"
  "claude-opus-4-7"
  "kimi-k2.5"
  "gemini-2.5-flash"
)

for model in "${MODELS[@]}"; do
  # Sanitize model name for filename
  safe_name=$(echo "$model" | tr '.' '-')
  outfile="$OUTDIR/${safe_name}_vissearch.json"
  logfile="$OUTDIR/${safe_name}_vissearch.log"

  # Skip if output already exists
  if [ -f "$outfile" ]; then
    echo "[SKIP] $model: $outfile already exists"
    continue
  fi

  echo "[START] $model -> $logfile"
  nohup python eval_multiturn.py \
    --benchmark pkc-test \
    --data-path "$DATA_PATH" \
    --search-backend local \
    --kb-path "$KB_PATH" \
    --api-base "$API_BASE" \
    --api-key "$API_KEY" \
    --model-name "$model" \
    --max-samples "$MAX_SAMPLES" \
    --max-turns 50 \
    --output "$outfile" \
    > "$logfile" 2>&1 &

  echo "  PID: $!"
  # Stagger starts by 5 seconds to avoid API rate limits
  sleep 5
done

echo ""
echo "All models launched. Monitor with:"
echo "  tail -f $OUTDIR/*.log"
echo ""
echo "Check progress:"
echo "  for f in $OUTDIR/*_vissearch.log; do echo \"=== \$(basename \$f) ===\"; tail -2 \$f; done"
