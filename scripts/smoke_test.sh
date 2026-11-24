#!/usr/bin/env bash
#
# smoke_test.sh — End-to-end check: generate one benchmark session, then
# evaluate a doctor-agent LLM on it. All LLM calls use DeepSeek V4 Flash.
#
# Prerequisites:
#   1. uv sync                       (install dependencies)
#   2. A repo-root .env containing:
#          DEEPSEEK_API_KEY=sk-...
#          DEEPSEEK_BASE_URL=https://api.deepseek.com
#      (The rubric judge falls back to DeepSeek when GATEWAY_OPENAI_API_KEY is unset,
#       so DEEPSEEK_API_KEY alone is enough to run this smoke test.)
#   3. MIMIC-IV patient data under raw_data/MIMIC-IV-split-by-patient/ — not in
#      this repository; see README and scripts/split_mimic_by_subject.py.
#
# Usage:
#   scripts/smoke_test.sh              # generate 1 session, then evaluate it
#   scripts/smoke_test.sh example      # skip generation; evaluate the example session
#
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root

MODE="${1:-generate}"
MODEL="deepseek-v4-flash"
# Default subject for the generation smoke test; its per-subject tables must be
# present under raw_data/ (see README). Override with SMOKE_SUBJECT_ID=<id>.
SUBJECT_ID="${SMOKE_SUBJECT_ID:-10458345}"
DIFFICULTY="${SMOKE_DIFFICULTY:-2}"

# Load DEEPSEEK_API_KEY from .env if not already exported
if [ -z "${DEEPSEEK_API_KEY:-}" ] && [ -f .env ]; then
  set -a; . ./.env; set +a
fi
if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
  echo "ERROR: DEEPSEEK_API_KEY is not set. Add it to .env at the repo root." >&2
  exit 1
fi

if [ "$MODE" = "example" ]; then
  DATA_DIR="physassistbench/data_smoke"
  mkdir -p "$DATA_DIR"
  EXAMPLE="physassistbench/examples/med_safety_example.jsonl"
  if [ ! -f "$EXAMPLE" ]; then
    echo "ERROR: $EXAMPLE not found. The example session is MIMIC-IV-derived and is" >&2
    echo "       distributed via PhysioNet, not in this repository (see README)." >&2
    exit 1
  fi
  cp "$EXAMPLE" "$DATA_DIR/med_safety.jsonl"
  echo ">> Using example session: $DATA_DIR/med_safety.jsonl (generation skipped)"
else
  DATA_DIR="physassistbench/data_smoke"
  echo ">> Step 1/2: generating one med_safety session for subject $SUBJECT_ID (DeepSeek V4 Flash) ..."
  uv run python physassistbench/generate_all.py \
    --scenarios med_safety --n 1 --subject_id "$SUBJECT_ID" \
    --difficulty "$DIFFICULTY" --out_dir "$DATA_DIR"
fi

echo ">> Step 2/2: evaluating $MODEL on the session ..."
uv run python physassistbench/run_eval.py \
  --model "$MODEL" --language en --skip_generate \
  --data_dir "$DATA_DIR" --output_dir physassistbench/results_smoke

echo ""
echo ">> Smoke test complete."
echo "   Data:    $DATA_DIR/med_safety.jsonl"
echo "   Results: physassistbench/results_smoke/ (eval_report.txt, metrics_summary.json)"
