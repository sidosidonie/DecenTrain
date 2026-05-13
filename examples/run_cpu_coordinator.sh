#!/usr/bin/env bash
# Run the multi-machine FFN example as the CPU *coordinator* (trusted side).
#
# Owns the SLALOM projection vectors (s / s_tilde), reproduces the input x on
# CPU from the shared seed, connects to the GPU worker, drives N forward
# rounds, and verifies every linear output the worker returns. Runs entirely
# on CPU — no GPU needed on this machine.
#
# Start examples/run_gpu_worker.sh on the GPU machine first, then run this with
# WORKER_HOST pointing at it.
#
# Usage:
#   WORKER_HOST=192.168.1.21 examples/run_cpu_coordinator.sh
#   WORKER_HOST=10.0.0.5 PORT=9100 ROUNDS=200 examples/run_cpu_coordinator.sh
#   WORKER_HOST=127.0.0.1 PIPELINE=1 examples/run_cpu_coordinator.sh   # if SSH-tunnelled
#
# Env knobs:
#   WORKER_HOST  GPU worker address                 (REQUIRED)
#   PORT         GPU worker TCP port                (default: 9100)
#   ROUNDS       forward rounds to run              (default: 100)
#   WARMUP       rounds excluded from the summary   (default: 10)
#   HIDDEN       FFN hidden size                    (default: 4096)
#   INTER        FFN intermediate size              (default: 11008)
#   BATCH        batch size                         (default: 1)
#   SEQ          sequence length                    (default: 512)
#   WIRE_DTYPE   fp16|fp32 on-wire activation dtype (default: fp16)
#   THRESHOLD    SLALOM MSE threshold               (default: auto: scales with dtype/dims)
#   JSON_REPORT  path for the per-round JSON report (default: ffn_run.json; empty string disables)
#   PIPELINE     1 to enable --pipeline             (default: 0)  -- must match the worker
#   VERBOSE      1 to print a line per round        (default: 0)
#   PYTHON       python interpreter                 (default: python)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ -z "${WORKER_HOST:-}" ]; then
  echo "error: set WORKER_HOST to the GPU worker's address, e.g.:" >&2
  echo "       WORKER_HOST=192.168.1.21 $0" >&2
  exit 2
fi

PORT="${PORT:-9100}"
ROUNDS="${ROUNDS:-100}"
WARMUP="${WARMUP:-10}"
HIDDEN="${HIDDEN:-4096}"
INTER="${INTER:-11008}"
BATCH="${BATCH:-1}"
SEQ="${SEQ:-512}"
WIRE_DTYPE="${WIRE_DTYPE:-fp16}"
PIPELINE="${PIPELINE:-0}"
VERBOSE="${VERBOSE:-0}"
PYTHON="${PYTHON:-python}"
JSON_REPORT="${JSON_REPORT-ffn_run.json}"   # note: ${VAR-default} so JSON_REPORT="" disables it

ARGS=(examples/multi_machine_ffn.py
      --role coordinator
      --worker-host "${WORKER_HOST}"
      --worker-port "${PORT}"
      --rounds "${ROUNDS}"
      --warmup "${WARMUP}"
      --hidden "${HIDDEN}"
      --inter "${INTER}"
      --batch "${BATCH}"
      --seq "${SEQ}"
      --wire-dtype "${WIRE_DTYPE}")
[ -n "${THRESHOLD:-}" ] && ARGS+=(--threshold "${THRESHOLD}")
[ -n "${JSON_REPORT}" ] && ARGS+=(--json-report "${JSON_REPORT}")
[ "${PIPELINE}" = "1" ] && ARGS+=(--pipeline)
[ "${VERBOSE}" = "1" ] && ARGS+=(--verbose)

echo "[coord] repo:    ${REPO_ROOT}"
echo "[coord] worker:  ${WORKER_HOST}:${PORT}"
echo "[coord] ffn:     SwiGLU hidden=${HIDDEN} inter=${INTER} batch=${BATCH} seq=${SEQ} wire=${WIRE_DTYPE}"
echo "[coord] rounds:  ${ROUNDS} (warmup ${WARMUP})   pipeline: ${PIPELINE}   report: ${JSON_REPORT:-<none>}"
echo "[coord] cmd:     ${PYTHON} ${ARGS[*]}"
echo

exec "${PYTHON}" "${ARGS[@]}"
