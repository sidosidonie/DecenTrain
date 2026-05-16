#!/usr/bin/env bash
#
# netspeed.sh — 测试两台 GCE 机器之间的内网网络速度(带宽 + 延迟)
#
# 用法:
#   机器 A(server 端):  ./netspeed.sh server
#   机器 B(client 端):  ./netspeed.sh client <A的内网IP>
#
# 依赖:iperf3(脚本会尝试自动安装)
# 防火墙:确保 VPC 防火墙允许两台机器间 TCP 5201,以及 ICMP(用于 ping)
#

set -uo pipefail

PORT="${PORT:-5201}"
PING_COUNT="${PING_COUNT:-20}"
IPERF_DURATION="${IPERF_DURATION:-20}"
PARALLEL_STREAMS="${PARALLEL_STREAMS:-8}"

# ---- 颜色 ----
if [[ -t 1 ]]; then
  BOLD=$'\033[1m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; CYAN=$'\033[36m'; RESET=$'\033[0m'
else
  BOLD=''; GREEN=''; YELLOW=''; CYAN=''; RESET=''
fi
info() { echo "${CYAN}==>${RESET} $*"; }
ok()   { echo "${GREEN}OK${RESET}  $*"; }
warn() { echo "${YELLOW}!!${RESET}  $*"; }

usage() {
  cat <<EOF
用法:
  $(basename "$0") server               # 在机器 A 上运行,启动 iperf3 服务端
  $(basename "$0") client <SERVER_IP>   # 在机器 B 上运行,连到机器 A 测速

可调环境变量(及当前值):
  PORT=$PORT  IPERF_DURATION=$IPERF_DURATION  PARALLEL_STREAMS=$PARALLEL_STREAMS  PING_COUNT=$PING_COUNT
EOF
  exit 1
}

# ---- 确保 iperf3 已安装 ----
ensure_iperf3() {
  command -v iperf3 >/dev/null 2>&1 && return
  info "未检测到 iperf3,尝试自动安装..."
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -qq && sudo apt-get install -y -qq iperf3
  elif command -v dnf >/dev/null 2>&1; then
    sudo dnf install -y -q iperf3
  elif command -v yum >/dev/null 2>&1; then
    sudo yum install -y -q iperf3
  else
    warn "无法自动安装 iperf3,请手动安装后重试"; exit 1
  fi
  ok "iperf3 安装完成"
}

# ---- 读取 GCE metadata(非 GCE 环境会静默跳过)----
md() {
  curl -s -m 2 -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/$1" 2>/dev/null || true
}

show_vm_info() {
  local mtype zone
  mtype=$(md "instance/machine-type" | awk -F/ '{print $NF}')
  zone=$(md "instance/zone" | awk -F/ '{print $NF}')

  echo "${BOLD}本机 GCE 信息${RESET}"
  if [[ -z "$mtype" ]]; then
    warn "非 GCE 环境或 metadata 不可用,跳过机器信息"
  else
    echo "  机器类型 : $mtype"
    echo "  Zone     : ${zone:-未知}"
    # 粗略估算理论带宽:约 2 Gbps/vCPU,未开 Tier_1 默认封顶 32 Gbps
    local vcpu cap
    vcpu=$(echo "$mtype" | grep -oE '[0-9]+$' || true)
    if [[ -n "$vcpu" ]]; then
      cap=$(( vcpu * 2 )); (( cap > 32 )) && cap=32
      echo "  vCPU     : ~${vcpu}(从机型名粗估,共享核机型不准)"
      echo "  理论带宽 : ~${cap} Gbps(未开 Tier_1 networking 的默认上限)"
    fi
  fi

  # 网卡驱动:gve = gVNIC(高带宽),virtio_net = 受限
  if command -v ethtool >/dev/null 2>&1; then
    local iface drv
    iface=$(ip route get 8.8.8.8 2>/dev/null | grep -oP 'dev \K\S+' || true)
    [[ -z "$iface" ]] && iface=$(ip -o link show 2>/dev/null | awk -F': ' '$2!="lo"{print $2; exit}')
    drv=$(ethtool -i "$iface" 2>/dev/null | awk '/^driver:/{print $2}')
    case "$drv" in
      gve)        echo "  网卡驱动 : gve (gVNIC) — 支持高带宽" ;;
      virtio_net) echo "  网卡驱动 : virtio_net — 高带宽受限,需 >10G 建议换 gVNIC" ;;
      *)          [[ -n "$drv" ]] && echo "  网卡驱动 : $drv" ;;
    esac
  fi
  echo
}

# ---- server 模式 ----
run_server() {
  ensure_iperf3
  show_vm_info
  info "在端口 $PORT 启动 iperf3 服务端,等待 client 连接..."
  info "client 端运行:  ./netspeed.sh client <本机内网IP>"
  info "按 Ctrl+C 停止"
  echo
  iperf3 -s -p "$PORT"
}

# ---- client 模式 ----
run_client() {
  local server_ip="${1:-}"
  [[ -z "$server_ip" ]] && usage
  ensure_iperf3
  show_vm_info

  echo "${BOLD}目标 server${RESET} : $server_ip"
  echo "================================================"

  # 1. 延迟
  info "1/4  延迟测试(ping x${PING_COUNT})..."
  if ! ping -c "$PING_COUNT" -q "$server_ip" 2>/dev/null | tail -n 2; then
    warn "ping 失败 — 可能 ICMP 被防火墙拦截,不影响下面的带宽测试"
  fi
  echo

  # 2. TCP 单流
  info "2/4  TCP 单流带宽(${IPERF_DURATION}s)..."
  if ! iperf3 -c "$server_ip" -p "$PORT" -t "$IPERF_DURATION" -f m \
        | grep -E 'sender|receiver'; then
    warn "iperf3 连接失败 — 确认 server 在运行,且防火墙放行 TCP $PORT"
    exit 1
  fi
  echo

  # 3. TCP 多流(逼近真实上限)
  info "3/4  TCP 多流带宽(${PARALLEL_STREAMS} 条并行流,${IPERF_DURATION}s)..."
  iperf3 -c "$server_ip" -p "$PORT" -t "$IPERF_DURATION" -P "$PARALLEL_STREAMS" -f m \
    | grep -E 'SUM.*(sender|receiver)' || warn "未取到 SUM 行"
  echo

  # 4. 反向(server -> client 方向)
  info "4/4  反向带宽(server→client 方向,${IPERF_DURATION}s)..."
  iperf3 -c "$server_ip" -p "$PORT" -t "$IPERF_DURATION" -P "$PARALLEL_STREAMS" -R -f m \
    | grep -E 'SUM.*(sender|receiver)' || warn "未取到 SUM 行"
  echo

  echo "================================================"
  ok "测试完成"
  cat <<EOF

${BOLD}结果怎么看:${RESET}
  - 延迟(rtt avg):同 zone 通常 ~0.1-0.3ms,跨 zone 同 region ~1ms 左右
  - 多流带宽:对照上面的"理论带宽"。接近上限说明网络没瓶颈;
    远低于上限则可能受机器规格 / virtio 网卡 / 未开 Tier_1 限制
  - 普通服务间通信:几 Gbps + 低延迟即够用
  - 分布式训练等高要求场景:需要同 zone + gVNIC + Tier_1 networking
EOF
}

# ---- 入口 ----
[[ $# -lt 1 ]] && usage
case "$1" in
  server) run_server ;;
  client) run_client "${2:-}" ;;
  *)      usage ;;
esac
