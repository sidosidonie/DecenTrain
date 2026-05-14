#!/usr/bin/env bash
# Launch N mini-zimage worker processes on THIS machine, one per port
# (PORT_BASE..PORT_BASE+N-1). Just delegates to run_zimage_worker.sh per
# worker. Pair with run_zimage_coordinator_fanout.sh on the other side.
#
# Usage:
#   examples/run_zimage_workers_fanout.sh                         # 5 workers
#   N=5 DEVICE=cuda examples/run_zimage_workers_fanout.sh         # round-robin all GPUs
#   FAULT=scale_o FAULT_BLOCK=3 examples/run_zimage_workers_fanout.sh
#   NO_STREAM=1 examples/run_zimage_workers_fanout.sh             # disable streaming sender
#
# Env knobs:
#   N            number of worker processes                       (default: 5)
#   PORT_BASE    first port                                       (default: 9103)
#   DEVICE       cpu | cuda | cuda:N | cuda:a,cuda:b,...          (default: cuda)
#   FAULT        none|flip_v|scale_o|scale_w2|flip_w1|drop_silu   (default: none)
#   FAULT_BLOCK  block index to inject the fault into             (default: 0)
#   NO_STREAM    1 to disable the streaming sender                (default: 0)
#   LOOP         1 = each worker restarts after a session         (default: 1)
#   DETACH       1 = launch workers and exit immediately          (default: 0)
#   THREADS      OMP/MKL threads per worker (CPU device)          (default: 8)
#   LOGDIR       dir for per-worker logs + pidfile                (default: workers_zimage_<ts>)
#   PYTHON       python interpreter                               (default: python)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

N="${N:-5}"
PORT_BASE="${PORT_BASE:-9103}"
DEVICE="${DEVICE:-cuda}"
FAULT="${FAULT:-none}"
FAULT_BLOCK="${FAULT_BLOCK:-0}"
NO_STREAM="${NO_STREAM:-0}"
LOOP="${LOOP:-1}"
DETACH="${DETACH:-0}"
THREADS="${THREADS:-8}"
PYTHON="${PYTHON:-python}"
LOGDIR="${LOGDIR:-workers_zimage_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOGDIR"

devices=()
case "$DEVICE" in
  cpu)
    for ((i = 0; i < N; i++)); do devices+=("cpu"); done
    ;;
  cuda)
    ngpu="$("$PYTHON" -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 0)"
    if [ "$ngpu" -lt 1 ]; then
      echo "error: DEVICE=cuda but torch.cuda.device_count()==0 on this machine." >&2
      echo "       Use DEVICE=cpu (CPU box) or DEVICE=cuda:0 to force." >&2
      exit 2
    fi
    for ((i = 0; i < N; i++)); do devices+=("cuda:$(( i % ngpu ))"); done
    echo "[workers] ${ngpu} GPU(s) visible -> round-robin assignment"
    ;;
  *,*)
    IFS=',' read -r -a pool <<< "$DEVICE"
    for ((i = 0; i < N; i++)); do devices+=("${pool[$(( i % ${#pool[@]} ))]}"); done
    ;;
  *)
    for ((i = 0; i < N; i++)); do devices+=("$DEVICE"); done
    ;;
esac

echo "[workers] launching ${N} mini-zimage worker(s)   ports ${PORT_BASE}..$((PORT_BASE + N - 1))   fault: ${FAULT}@b${FAULT_BLOCK}   stream: $([ "${NO_STREAM}" = "1" ] && echo off || echo on)   loop: ${LOOP}"
echo "[workers] devices:  ${devices[*]}"
echo "[workers] logs:     ${LOGDIR}/   (pidfile: ${LOGDIR}/pids)"
echo

: > "${LOGDIR}/pids"
pids=()
for ((i = 0; i < N; i++)); do
  port=$((PORT_BASE + i))
  dev="${devices[$i]}"
  env PORT="$port" DEVICE="$dev" FAULT="$FAULT" FAULT_BLOCK="$FAULT_BLOCK" \
      NO_STREAM="$NO_STREAM" LOOP="$LOOP" PYTHON="$PYTHON" \
      OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
      bash examples/run_zimage_worker.sh > "${LOGDIR}/worker_${i}.log" 2>&1 &
  pid=$!
  pids+=("$pid")
  echo "$pid port=$port device=$dev" >> "${LOGDIR}/pids"
  echo "[workers] worker ${i}: ${dev}  ->  0.0.0.0:${port}   pid ${pid}   log ${LOGDIR}/worker_${i}.log"
done

sleep 2
echo
if command -v ss >/dev/null 2>&1; then
  echo "[workers] listening ports:"
  ss -ltn 2>/dev/null | awk -v lo="$PORT_BASE" -v hi="$((PORT_BASE + N - 1))" \
    'NR>1 { n=split($4,a,":"); p=a[n]; if (p+0>=lo && p+0<=hi) print "    " $4 }' \
    | sort -u || true
fi
echo

stop_all() {
  echo
  echo "[workers] stopping ${#pids[@]} worker(s)..."
  for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done
  wait 2>/dev/null || true
  echo "[workers] stopped."
}

if [ "$DETACH" = "1" ]; then
  echo "[workers] DETACH=1 — workers left running in the background."
  echo "[workers] stop them with:  kill \$(awk '{print \$1}' ${LOGDIR}/pids)"
  echo "[workers]            or:   pkill -f 'multi_machine_zimage.py --role worker'"
  exit 0
else
  trap stop_all INT TERM EXIT
  echo "[workers] running in foreground — press Ctrl-C to stop all ${N} workers."
  wait
fi
