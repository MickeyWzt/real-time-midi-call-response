#!/bin/zsh
set -euo pipefail

PROJECT_DIR="${0:A:h}"
cd "$PROJECT_DIR"

if [[ ! -x .venv/bin/python ]]; then
  print -u2 "Missing .venv. Complete the project setup before starting the demo."
  exit 1
fi

export HF_HOME="$PROJECT_DIR/hf_cache"
export HF_HUB_CACHE="$PROJECT_DIR/hf_cache/hub"
export HF_HUB_DISABLE_XET=1

mkdir -p logs

DEMO_URL="http://127.0.0.1:8000/"
SESSION_NAME="mfp-interface"

if curl --silent --fail "$DEMO_URL" >/dev/null 2>&1; then
  print "MFP is already running. Opening the control surface."
  if [[ "${MFP_SKIP_OPEN:-0}" != "1" ]]; then
    open "$DEMO_URL"
  fi
  exit 0
fi

screen -S "$SESSION_NAME" -X quit >/dev/null 2>&1 || true
screen -wipe >/dev/null 2>&1 || true
screen -dmS "$SESSION_NAME" .venv/bin/python code/interface_backend.py --host 127.0.0.1 --port 8000

for attempt in {1..50}; do
  if curl --silent --fail "$DEMO_URL" >/dev/null 2>&1; then
    print "MFP is ready. The background service will remain active."
    if [[ "${MFP_SKIP_OPEN:-0}" != "1" ]]; then
      open "$DEMO_URL"
    fi
    exit 0
  fi
  sleep 0.1
done

print -u2 "MFP could not start. Run this file again, or inspect the project logs."
exit 1
