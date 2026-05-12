# AIGC Rewriter

基于 Ollama 和 `skskk/aigc-rewriter` GGUF 模型的 OpenAI 兼容降 AIGC 文本改写服务。

当前线上服务：

- 服务地址：`http://117.50.89.11`
- OpenAI base_url：`http://117.50.89.11/v1`
- 客户端 model：`local`
- Ollama 模型：`qwen3-aigc:latest`
- GGUF 来源：`https://huggingface.co/skskk/aigc-rewriter/resolve/main/qwen3-merged-aigc_zhv3-Q4_K_M.gguf`

## 功能

- 传入一段中文文本，返回降 AIGC 改写后的正文。
- 提供 OpenAI 兼容接口：`/v1/chat/completions`、`/v1/models`。
- 使用 Ollama `raw: true` 调用 GGUF completion 模型，服务端使用 minimal prompt 控制输入输出。
- 只保留并使用 `qwen3-aigc:latest`，不依赖其它 Ollama 模型。

## 安装

```bash
uv sync
```

安装 Ollama 后，下载 GGUF：

```bash
mkdir -p models
curl -L -o models/qwen3-merged-aigc_zhv3-Q4_K_M.gguf \
  https://huggingface.co/skskk/aigc-rewriter/resolve/main/qwen3-merged-aigc_zhv3-Q4_K_M.gguf
```

导入 Ollama：

```bash
cat > Modelfile.qwen3-aigc <<'EOF'
FROM ./models/qwen3-merged-aigc_zhv3-Q4_K_M.gguf
PARAMETER temperature 0.7
PARAMETER num_ctx 4096
EOF

ollama create qwen3-aigc -f Modelfile.qwen3-aigc
ollama list
```

如果只允许使用该模型，删除其它模型：

```bash
ollama rm qwen2.5:1.5b || true
ollama rm deepseek-r1:1.5b || true
```

## 启动

```bash
OLLAMA_MODEL=qwen3-aigc:latest SERVER_PORT=8000 uv run aigc_rewriter_server.py
```

环境变量：

| 变量 | 默认值 | 说明 |
|---|---:|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama 地址 |
| `OLLAMA_MODEL` | `qwen3-aigc:latest` | Ollama 模型名 |
| `SERVER_HOST` | `0.0.0.0` | API 监听地址 |
| `SERVER_PORT` | `8000` | API 监听端口 |
| `OLLAMA_KEEP_ALIVE` | `24h` | Ollama 模型常驻时间 |
| `SERVER_MIN_TOKENS` | `128` | 服务端最小输出 token 预算 |
| `SERVER_MAX_TOKENS` | `512` | 服务端最大输出 token 上限 |
| `SERVER_MAX_TEMPERATURE` | `0.45` | 服务端最大温度上限 |
| `SENTENCES_PER_CALL` | `5` | 每次送入模型的句子数 |
| `CHARS_PER_CALL` | `800` | 单块最大字符数 |
| `AIGC_REWRITE_INSTRUCTION` | 内置 minimal 指令 | 降 AIGC prompt 前缀 |

## OpenAI 格式使用

curl：

```bash
curl -X POST http://117.50.89.11/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [
      {
        "role": "user",
        "content": "人工智能技术的发展正在深刻改变人类社会的生产方式和生活方式。"
      }
    ]
  }'
```

Python OpenAI SDK：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://117.50.89.11/v1",
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="local",
    messages=[
        {
            "role": "user",
            "content": "人工智能技术的发展正在深刻改变人类社会的生产方式和生活方式。",
        }
    ],
)

print(response.choices[0].message.content)
```

## 接口

- `GET /`：健康检查，返回 Ollama host、当前模型、可用模型。
- `GET /v1/models`：OpenAI 兼容模型列表。
- `POST /v1/chat/completions`：OpenAI 兼容聊天补全格式，实际执行降 AIGC 改写。

注意：`stream: true` 当前不会返回 SSE 流，服务始终返回普通 JSON。

## 性能策略

为了配合客户端可能传入的 `max_tokens=2048`、`temperature=0.7`、多轮改写等配置，服务端会做保护：

- `max_tokens` 会按输入长度自适应，默认范围是 `128-512`。
- `temperature` 会被限制到 `SERVER_MAX_TEMPERATURE`，默认 `0.45`。
- 默认每 `5` 句送入一次模型，优先保证整体改写效果。
- Ollama 请求带 `keep_alive=24h`，模型保持 GPU 常驻，避免每次冷加载。
- 服务端只取最后一条 `user` 消息作为待改写文本，避免格式上下文拖慢 GGUF completion 模型。
