#!/bin/bash
###############################################################################
# KsanaLLM 快速上手脚本 (Iluvatar MR-V100)
# 在容器 yzc_iluvatar_ksanallm_work 内直接运行
#
# 用法:
#   bash quickstart.sh build       # cmake + make 编译
#   bash quickstart.sh server      # 启动推理服务
#   bash quickstart.sh test        # 发送测试请求
#   bash quickstart.sh all         # build -> server(后台) -> test
#
# 环境变量:
#   GPU_ID            GPU 编号                       (默认: 0)
#   PORT              服务端口                       (默认: 8080)
#   CONFIG            yaml 配置文件                  (默认: examples/ksana_llm2-7b.yaml)
#   JOBS              编译并行度                     (默认: 16)
#   KLLM_LOG_LEVEL    运行时日志级别 INFO|DEBUG|...  (默认: INFO; 设为 DEBUG 开启 [DBG-*] 打印)
###############################################################################

set -euo pipefail

GPU_ID=${GPU_ID:-0}
PORT=${PORT:-8080}
JOBS=${JOBS:-16}

KSANA_ROOT=/home/zhichao.yang/KsanaLLM
CONFIG=${CONFIG:-${KSANA_ROOT}/examples/ksana_llm2-7b.yaml}
PYTHON_DIR=${KSANA_ROOT}/src/ksana_llm/python

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] WARN:${NC} $*"; }
err()  { echo -e "${RED}[$(date +%H:%M:%S)] ERROR:${NC} $*"; exit 1; }

###############################################################################
# 1. 编译
###############################################################################
do_build() {
    log "=== 编译 KsanaLLM (WITH_ILUVATAR=ON) ==="

    mkdir -p "${KSANA_ROOT}/build"
    cd "${KSANA_ROOT}/build"

    if [ ! -f CMakeCache.txt ]; then
        log "cmake 配置..."
        cmake -DWITH_ILUVATAR=ON -DWITH_CUDA=OFF -DWITH_TESTING=OFF ..
    else
        log "CMakeCache 已存在，跳过 cmake 配置（如需重新配置请先 rm -rf build/）"
    fi

    log "make -j${JOBS}，耗时较长请耐心等待..."
    make -j${JOBS} torch_serving loguru

    # 验证产物
    [ -f "${KSANA_ROOT}/build/lib/libtorch_serving.so" ] \
        || err "编译失败：libtorch_serving.so 未生成"

    # 建立 python/lib 软链接
    ln -sfn "${KSANA_ROOT}/build/lib" "${PYTHON_DIR}/lib"

    log "编译完成！产物:"
    ls -lh "${KSANA_ROOT}/build/lib/"*.so
}

###############################################################################
# 2. 启动 Server
###############################################################################
do_server() {
    log "=== 启动 KsanaLLM Server ==="
    log "GPU: ${GPU_ID} | 端口: ${PORT} | 配置: ${CONFIG} | 日志级别: ${KLLM_LOG_LEVEL:-INFO}"

    [ -L "${PYTHON_DIR}/lib" ] || [ -d "${PYTHON_DIR}/lib" ] \
        || err "python/lib 不存在，请先执行 bash quickstart.sh build"

    [ -f "${CONFIG}" ] \
        || err "配置文件不存在: ${CONFIG}"

    export CUDA_VISIBLE_DEVICES=${GPU_ID}

    log "启动中... (Ctrl+C 停止)；如需调试日志: KLLM_LOG_LEVEL=DEBUG bash quickstart.sh server"
    cd "${PYTHON_DIR}"
    python3 serving_server.py \
        --config_file "${CONFIG}" \
        --port ${PORT}
}

# 后台启动 (给 do_all 用)
do_server_bg() {
    log "后台启动 KsanaLLM Server..."

    fuser -k -9 ${PORT}/tcp 2>/dev/null || true
    sleep 1

    export CUDA_VISIBLE_DEVICES=${GPU_ID}
    cd "${PYTHON_DIR}"
    python3 serving_server.py \
        --config_file "${CONFIG}" \
        --port ${PORT} \
        > "${KSANA_ROOT}/server.log" 2>&1 &
    disown

    log "等待 server 就绪 (日志: ${KSANA_ROOT}/server.log)..."
    for i in $(seq 1 60); do
        if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
            log "Server 已就绪! (http://localhost:${PORT})"
            return 0
        fi
        sleep 5
    done
    err "Server 启动超时 (5 分钟)，查看日志: tail -50 ${KSANA_ROOT}/server.log"
}

###############################################################################
# 3. 发送测试请求
###############################################################################
do_test() {
    log "=== 发送测试请求 (localhost:${PORT}) ==="

    log "--- curl /generate ---"
    curl -s --max-time 60 "http://127.0.0.1:${PORT}/generate" \
        -H "Content-Type: application/json" \
        -d '{"prompt":"Hello, what is your name?","sampling_config":{"temperature":0.0,"topk":1,"max_new_tokens":16},"stream":false}' || true
    echo ""

    log "--- curl /v1/chat/completions (OpenAI 兼容) ---"
    curl -s --max-time 60 "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"ksana-llm","messages":[{"role":"user","content":"Hello"}],"max_tokens":32,"temperature":0.0}' || true
    echo ""
}

###############################################################################
# 4. 全流程
###############################################################################
do_all() {
    do_build
    do_server_bg
    do_test
    log "全流程完成!"
}

###############################################################################
# 入口
###############################################################################
usage() {
    cat <<'USAGE'
用法: bash quickstart.sh <command>

Commands:
  build     cmake + make 编译 (WITH_ILUVATAR=ON)
  server    前台启动推理服务
  test      发送测试请求
  all       全流程: build -> server(后台) -> test

环境变量:
  GPU_ID=0                               GPU 编号
  PORT=8080                              服务端口
  CONFIG=examples/ksana_llm2-7b.yaml     yaml 配置文件路径
  JOBS=16                                编译并行度

示例:
  # 编译
  bash quickstart.sh build

  # GPU 1, 端口 9090 启动
  GPU_ID=1 PORT=9090 bash quickstart.sh server

  # 用 Qwen2 配置
  CONFIG=/home/zhichao.yang/KsanaLLM/examples/ksana_llm_qwen2.yaml bash quickstart.sh server

  # 发请求
  bash quickstart.sh test

  # 直接 curl
  curl http://localhost:8080/generate \
    -H 'Content-Type: application/json' \
    -d '{
      "prompt": "你好，请介绍一下自己",
      "sampling_config": {
        "temperature": 0.0,
        "topk": 1,
        "max_new_tokens": 64,
        "repetition_penalty": 1.0
      },
      "stream": false
    }'
USAGE
}

case "${1:-}" in
    build)  do_build ;;
    server) do_server ;;
    test)   do_test ;;
    all)    do_all ;;
    *)      usage ;;
esac
