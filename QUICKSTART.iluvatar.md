下面是适配后的 Iluvatar (天数 Corex MR-V100) 后端在容器内从零跑通 KsanaLLM 的最小步骤。所有路径与命令均针对当前仓库与容器环境。

## 0. 前提

| 项 | 值 |
|---|---|
| 硬件 | Iluvatar Corex MR-V100 (BF16 支持) |
| 容器 | `yzc_iluvatar_ksanallm_work` (镜像 `iluvatar_common/vllm0.17.0-4.4.0-x86:v3`) |
| 仓库 | `/home/zhichao.yang/KsanaLLM`，宿主机与容器同路径挂载 |
| 分支 | `Iluvatar_adapt`（HEAD ≥ `b116b81c`）|
| 模型 | `/home/zhichao.yang/KsanaLLM/src/ksana_llm/python/TinyLlama-1.1B-Chat`（含 `model.safetensors`） |
| 配置 | `examples/ksana_llm2-7b.yaml`（`model_dir` 已指向上述模型） |

启动容器（如已 Up 可跳过）：
```bash
docker exec -it yzc_iluvatar_ksanallm_work bash
cd /home/zhichao.yang/KsanaLLM
```

确认分支与 commit：
```bash
git checkout Iluvatar_adapt
git log -1 --oneline
# b116b81c fix(iluvatar): make TinyLlama generate human-readable text
```

---

## 1. 一键全流程（推荐）

```bash
GPU_ID=0 PORT=8080 bash quickstart.sh all
```

`do_all` 会依次执行：build → 后台启动 server → 自动 curl `/generate` 与 `/v1/chat/completions` 测试。

如果只想分步执行，见下面三步。

---

## 2. 分步操作

### 2.1 编译

```bash
bash quickstart.sh build
```

等价于：
```bash
mkdir -p build && cd build
cmake -DWITH_ILUVATAR=ON -DWITH_CUDA=OFF -DWITH_TESTING=OFF ..
make -j16 torch_serving loguru
ln -sfn /home/zhichao.yang/KsanaLLM/build/lib \
        /home/zhichao.yang/KsanaLLM/src/ksana_llm/python/lib
```

预期产物：
```
build/lib/libtorch_serving.so
build/lib/libloguru.so
```

> 想完全重建：`rm -rf build/` 后重跑。增量编译几十秒到几分钟。

### 2.2 启动 Server

前台：
```bash
GPU_ID=0 PORT=8080 bash quickstart.sh server
```

后台 + 日志：
```bash
GPU_ID=0 PORT=8080 nohup bash quickstart.sh server > server.log 2>&1 &
tail -f server.log | grep -E "Uvicorn|ERROR|Traceback"
```

就绪标志：
```
INFO: Uvicorn running on http://0.0.0.0:8080
```

### 2.3 测试请求

一键：
```bash
bash quickstart.sh test
```

手动多 prompt 验证：
```bash
for p in 'Hello, how are you?' 'The capital of France is' 'Once upon a time' '1+1='; do
  echo "=== Prompt: $p ==="
  curl -s -m 60 -X POST http://localhost:8080/generate \
    -H 'Content-Type: application/json' \
    -d "{\"prompt\": \"$p\", \"sampling_config\": {\"temperature\": 0.0, \"topk\": 1, \"topp\": 0.0, \"max_new_tokens\": 30, \"repetition_penalty\": 1.0}}" \
    | python3 -c 'import sys,json,re; raw=sys.stdin.read().strip(); objs=re.findall(r"\{[^}]*\"texts\"[^}]*\}", raw); d=json.loads(objs[-1]); print("OUT:", repr(d["texts"][0]))'
done
```

预期输出：
```
=== Prompt: Hello, how are you? ===
OUT: '\nI am fine, thank you.\nHow are you? I am fine, thank you.\nCan you repeat that again?\nCan you'
=== Prompt: The capital of France is ===
OUT: 'Paris.\n\n2. B. The capital of Germany is Berlin.\n\n3. C. The capital of the United States is Washington,'
=== Prompt: Once upon a time ===
OUT: ", there was a young woman named Lily. She lived in a small town, where everyone knew each other's names. Lily was a"
=== Prompt: 1+1= ===
OUT: '2\n- 2 + 2 = 4\n- 4 + 4 = 8\n- 8 + 8 = '
```

> **接口注意**：必须用 `prompt`（单数）+ `sampling_config`，不能用 `prompts` + `sampling_params`，否则 200 OK 但返回空。返回是 streaming，多个 JSON object 拼接，解析时取最后一个。

OpenAI 兼容接口：
```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"ksana-llm","messages":[{"role":"user","content":"Hello"}],"max_tokens":32,"temperature":0.0}'
```

---

## 3. 与 HuggingFace 对齐验证（可选）

```bash
python3 - <<'PY'
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
PATH = '/home/zhichao.yang/KsanaLLM/src/ksana_llm/python/TinyLlama-1.1B-Chat'
tk = AutoTokenizer.from_pretrained(PATH)
m  = AutoModelForCausalLM.from_pretrained(PATH, torch_dtype=torch.bfloat16).cuda()
for p in ['Hello, how are you?', 'The capital of France is', 'Once upon a time', '1+1=']:
    inp = tk(p, return_tensors='pt').to('cuda')
    out = m.generate(**inp, max_new_tokens=30, do_sample=False)
    new = out[0][inp.input_ids.shape[1]:]
    print(f'=== HF: {p!r} ===')
    print('TEXT:', repr(tk.decode(new, skip_special_tokens=True)))
PY
```

判定标准：
- `Once upon a time` 与 Ksana **完全一致**（30 tokens bit-exact）。
- 其他 prompt 前若干 token 一致后分叉，但都是合理英文。这是 BF16 GEMM reduce 顺序差异，不是 bug。

---

## 4. 停止服务

```bash
# 前台启动: Ctrl+C
# 后台启动:
pkill -f serving_server.py
fuser -k -9 8080/tcp 2>/dev/null || true
```

---

## 5. 常见故障速查

| 现象 | 原因 | 修复 |
|---|---|---|
| `ModuleNotFoundError: libtorch_serving` | `python/lib` 软链未建 | 重跑 `bash quickstart.sh build` |
| `8080` 端口被占 | 上次 server 没 kill | `fuser -k -9 8080/tcp` 或 `pkill -f serving_server.py` |
| `/generate` 返回 `{"texts":[""]}` | 请求体字段名错（`prompts`/`sampling_params`） | 改为 `prompt`/`sampling_config` |
| 输出 `948,6552` 死循环 / `allocate це...` | 拉到了旧 commit | `git checkout Iluvatar_adapt && git pull`（HEAD ≥ `b116b81c`） |
| GPU 0 被占 | 换 GPU | `GPU_ID=1 bash quickstart.sh server` |
| yaml 报 `model_dir not found` | 模型未下载 | 从 HF 拉 `TinyLlama/TinyLlama-1.1B-Chat-v1.0` 到 `src/ksana_llm/python/TinyLlama-1.1B-Chat` |
| 编译找不到 `cuinfer.h` / `ixinfer.h` | 容器 SDK 路径不一致 | 检查 `cmake/iluvatar.cmake` 里的 SDK 根目录 |

---

## 6. 关键环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `GPU_ID` | `0` | `CUDA_VISIBLE_DEVICES` |
| `PORT` | `8080` | server 端口 |
| `CONFIG` | `examples/ksana_llm2-7b.yaml` | 模型/调度 yaml |
| `JOBS` | `16` | `make -j` 并行度 |
| `KLLM_LOG_LEVEL` | `INFO` | ksana 自带日志级别；设 `DEBUG` 启用 `[DBG-*]` 打印（运行时切换，无需重编） |

示例：
```bash
GPU_ID=1 PORT=9090 CONFIG=/abs/path/to/other.yaml bash quickstart.sh server

# 默认运行（仅 INFO 及以上）
bash quickstart.sh server

# 临时开启 bring-up 调试日志（写入 log/ksana_llm.log）
KLLM_LOG_LEVEL=DEBUG bash quickstart.sh server
```

---

## 7. 适配涉及的关键路径（debug 时定位用）

| 路径 | 作用 |
|---|---|
| `cmake/iluvatar.cmake` | `WITH_ILUVATAR=ON` 时的 toolchain / SDK 路径 |
| `src/ksana_llm/utils/iluvatar/` | Device / Context（对应 nvidia 同名目录） |
| `src/ksana_llm/kernels/iluvatar/cuda_kernels.cu` | BF16 / FP16 自写 kernel（SiLU、SwiGLU、Embedding、FusedAddRmsNorm 等） |
| `src/ksana_llm/layers/iluvatar/llm_kernels_stubs.cpp` | 与 NVIDIA `LLM_kernels` 对齐的 op 实现（`InvokeGatedActivation`、`InvokeFusedAddRmsNorm`、`InvokeStridedBatchedMatMul` 等） |
| `src/ksana_llm/layers/iluvatar/layer_stubs.cpp` | NVIDIA 专属 layer 的最小实现（CustomAllReduce 等 noop） |
| `3rdparty/LLM_kernels/csrc/kernels/iluvatar/` | paged attention / RoPE / cache_copy 的天数移植版 |
