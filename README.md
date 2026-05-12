# AIGC Rewriter

基于 Ollama 和 `skskk/aigc-rewriter` GGUF 模型的 OpenAI 兼容降 AIGC 文本改写服务。

当前线上服务：

- 服务地址：`http://117.50.89.11`
- OpenAI base_url：`http://117.50.89.11/v1`
- 客户端 model：`local`
- Ollama 模型：`qwen3-aigc-chat:latest`
- GGUF 来源：`https://huggingface.co/skskk/aigc-rewriter/resolve/main/qwen3-merged-aigc_zhv3-Q4_K_M.gguf`

## 功能

- 传入一段中文文本，返回降 AIGC 改写后的正文。
- 提供 OpenAI 兼容接口：`/v1/chat/completions`、`/v1/models`。
- 使用 Ollama `/api/chat` 调用显式 ChatML 模板，尽量复刻旧版 llama-cpp `create_chat_completion` 行为。
- 只保留并使用 `qwen3-aigc-chat:latest`，不依赖其它 Ollama 模型。

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
cat > Modelfile.qwen3-aigc-chat <<'EOF'
FROM ./models/qwen3-merged-aigc_zhv3-Q4_K_M.gguf
TEMPLATE """{{- range .Messages }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{- end }}<|im_start|>assistant
<think>

</think>

"""
PARAMETER temperature 0.7
PARAMETER num_ctx 4096
PARAMETER stop <|im_end|>
PARAMETER stop <|endoftext|>
EOF

ollama create qwen3-aigc-chat -f Modelfile.qwen3-aigc-chat
ollama list
```

如果只允许使用该模型，删除其它模型：

```bash
ollama rm qwen2.5:1.5b || true
ollama rm deepseek-r1:1.5b || true
ollama rm qwen3-aigc:latest || true
```

## 启动

```bash
OLLAMA_MODEL=qwen3-aigc-chat:latest SERVER_PORT=8000 uv run aigc_rewriter_server.py
```

环境变量：

| 变量 | 默认值 | 说明 |
|---|---:|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama 地址 |
| `OLLAMA_MODEL` | `qwen3-aigc-chat:latest` | Ollama 模型名 |
| `SERVER_HOST` | `0.0.0.0` | API 监听地址 |
| `SERVER_PORT` | `8000` | API 监听端口 |
| `OLLAMA_KEEP_ALIVE` | `24h` | Ollama 模型常驻时间 |

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

## 迁移说明

旧版 `10bfc5a41c322340157cc05d80871cb495c195ac` 使用 llama-cpp `create_chat_completion(messages=...)`，会读取 GGUF 内置 `tokenizer.chat_template`。Ollama 导入 GGUF 时默认生成的 `TEMPLATE {{ .Prompt }}` 不等价，会导致改写效果明显下降。当前 Modelfile 显式加入 Qwen3 ChatML 模板，并让服务端把 OpenAI `messages` 原样传给 `/api/chat`。
