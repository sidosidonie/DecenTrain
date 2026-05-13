#!/usr/bin/env bash
# Run N CPU coordinators in parallel against ONE worker host that is running N
# worker processes on consecutive ports (PORT_BASE .. PORT_BASE+N-1).
#
# The FFN example is 1 coordinator <-> 1 worker per TCP session, so "5 workers
# on one machine" means: launch 5 worker processes there, each on its own port
# (PORT=9100 LOOP=1 ..., PORT=9101 LOOP=1 ..., ...), then run this on the
# coordinator side to drive all 5 at once and aggregate the per-stream reports.
#
# Worker side (on the worker machine), e.g.:
#   for i in 0 1 2 3 4; do
#     PORT=$((9100+i)) DEVICE=cuda:0 LOOP=1 examples/run_gpu_worker.sh \
#         > /tmp/worker_$i.log 2>&1 &
#   done
#   # one GPU shared by all 5 -> they partly serialize on the device.
#   # multiple GPUs: DEVICE=cuda:$i. CPU-only box: DEVICE=cpu (cores contend).
#
# Usage:
#   WORKER_HOST=10.128.0.3 examples/run_cpu_coordinator_fanout.sh
#   WORKER_HOST=10.128.0.3 N=5 PORT_BASE=9100 ROUNDS=100 examples/run_cpu_coordinator_fanout.sh
#
# Env knobs:
#   WORKER_HOST  worker address                         (REQUIRED)
#   N            number of parallel coordinators        (default: 5)
#   PORT_BASE    first worker port; uses PORT_BASE..+N-1 (default: 9100)
#   ROUNDS       forward rounds per stream              (default: 100)
#   WARMUP       rounds excluded from each summary      (default: 10)
#   HIDDEN       FFN hidden size                        (default: 4096)
#   INTER        FFN intermediate size                  (default: 11008)
#   BATCH        batch size                             (default: 1)
#   SEQ          sequence length                        (default: 512)
#   WIRE_DTYPE   fp16|fp32 on-wire activation dtype     (default: fp16)
#   LINK_GBPS    nominal link bandwidth (Gbit/s)        (default: unset)
#   PIPELINE     1 to enable --pipeline on every coord  (default: 0)  -- must match the workers
#   THREADS      OMP/torch threads per coordinator      (default: 8)
#   OUTDIR       dir for per-stream logs + JSON reports (default: fanout_<ts>)
#   PYTHON       python interpreter                     (default: python)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ -z "${WORKER_HOST:-}" ]; then
  echo "error: set WORKER_HOST to the worker's address, e.g.:" >&2
  echo "       WORKER_HOST=10.128.0.3 $0" >&2
  exit 2
fi

N="${N:-5}"
PORT_BASE="${PORT_BASE:-9100}"
ROUNDS="${ROUNDS:-100}"
WARMUP="${WARMUP:-10}"
HIDDEN="${HIDDEN:-4096}"
INTER="${INTER:-11008}"
BATCH="${BATCH:-1}"
SEQ="${SEQ:-512}"
WIRE_DTYPE="${WIRE_DTYPE:-fp16}"
PIPELINE="${PIPELINE:-0}"
THREADS="${THREADS:-8}"
PYTHON="${PYTHON:-python}"
OUTDIR="${OUTDIR:-fanout_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTDIR"

echo "[fanout] worker:   ${WORKER_HOST}  ports ${PORT_BASE}..$((PORT_BASE+N-1))"
echo "[fanout] streams:  ${N}   ffn: SwiGLU hidden=${HIDDEN} inter=${INTER} batch=${BATCH} seq=${SEQ} wire=${WIRE_DTYPE}"
echo "[fanout] rounds:   ${ROUNDS} (warmup ${WARMUP})   threads/stream: ${THREADS}   outdir: ${OUTDIR}"
echo

pids=()
for i in $(seq 0 $((N-1))); do
  port=$((PORT_BASE + i))
  ARGS=(examples/multi_machine_ffn.py
        --role coordinator
        --worker-host "${WORKER_HOST}" --worker-port "${port}"
        --rounds "${ROUNDS}" --warmup "${WARMUP}"
        --hidden "${HIDDEN}" --inter "${INTER}" --batch "${BATCH}" --seq "${SEQ}"
        --wire-dtype "${WIRE_DTYPE}"
        --json-report "${OUTDIR}/coord_${i}.json")
  [ -n "${THRESHOLD:-}" ] && ARGS+=(--threshold "${THRESHOLD}")
  [ -n "${LINK_GBPS:-}" ] && ARGS+=(--link-gbps "${LINK_GBPS}")
  [ "${PIPELINE}" = "1" ] && ARGS+=(--pipeline)
  OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" \
    "${PYTHON}" "${ARGS[@]}" > "${OUTDIR}/coord_${i}.log" 2>&1 &
  pids+=($!)
  echo "[fanout] stream ${i} -> ${WORKER_HOST}:${port}   pid ${pids[-1]}   log ${OUTDIR}/coord_${i}.log"
done

rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo
echo "[fanout] all ${N} streams finished (rc=${rc}); per-stream logs + reports in ${OUTDIR}/"
echo

"${PYTHON}" - "${OUTDIR}" <<'PY'
import glob, json, os, sys
d = sys.argv[1]
files = sorted(glob.glob(os.path.join(d, "coord_*.json")))
if not files:
    print("(no JSON reports found)"); sys.exit(0)
rows = []
for f in files:
    s = json.load(open(f))["summary"]
    rows.append((os.path.basename(f), s))
print(f"=== fan-out summary: {len(rows)} parallel streams ===\n")
print(f"{'stream':<14}{'round/s':>9}{'p50 ms':>9}{'p95 ms':>9}{'wire ms':>9}{'verify ms':>11}{'passed':>9}")
agg_rps = agg_pass = agg_total = 0.0
for name, s in rows:
    rps = s["mean_round_per_s"]; agg_rps += rps
    agg_pass += s["rounds_passed"]; agg_total += s["rounds_measured"]
    print(f"{name:<14}{rps:>9.2f}{s['p50_end_to_end_ms']:>9.1f}{s['p95_end_to_end_ms']:>9.1f}"
          f"{s['mean_wire_recv_ms']:>9.1f}{s['mean_cpu_verify_ms']:>11.1f}"
          f"{s['rounds_passed']:>5}/{s['rounds_measured']:<3}")
print(f"\naggregate throughput  ≈ {agg_rps:.2f} round/s  (sum over {len(rows)} streams)")
print(f"verification          {int(agg_pass)}/{int(agg_total)} rounds passed across all streams")
# bandwidth, if recorded
effs = [s.get("mean_wire_effective_gbps") for _, s in rows if s.get("mean_wire_effective_gbps")]
if effs:
    print(f"per-stream effective  {', '.join(f'{x:.2f}' for x in effs)} Gbit/s"
          f"   (sum ≈ {sum(effs):.2f} Gbit/s)")
PY
