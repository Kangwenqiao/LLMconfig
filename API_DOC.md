# AIGC Rewriter API 文档

## 基本信息

- 服务地址：`http://117.50.89.11`
- OpenAI base_url：`http://117.50.89.11/v1`
- 客户端 model：`local`
- Ollama 模型：`qwen3-aigc-chat:latest`
- GGUF 模型：`qwen3-merged-aigc_zhv3-Q4_K_M.gguf`
- 模型来源：`skskk/aigc-rewriter`
- 推理方式：Ollama `/api/chat`，显式 Qwen3 ChatML 模板

## 健康检查

```bash
curl http://117.50.89.11/
```

响应示例：

```json
{
  "status": "ok",
  "ollama_host": "http://localhost:11434",
  "ollama_model": "qwen3-aigc-chat:latest",
  "available_models": ["qwen3-aigc-chat:latest"]
}
```

## 模型列表

```bash
curl http://117.50.89.11/v1/models
```

响应示例：

```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen3-aigc-chat:latest",
      "object": "model",
      "created": 1778565202,
      "owned_by": "ollama"
    }
  ]
}
```

## 降 AIGC 改写

请求：

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
    ],
    "temperature": 0.7,
    "max_tokens": 2048
  }'
```

响应：

```json
{
  "id": "chatcmpl-1778565612",
  "object": "chat.completion",
  "created": 1778565612,
  "model": "local",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "人工智能技术的发展正以前所未有的速度和规模改变着人类社会的生产方式和生活方式。"
      },
      "finish_reason": "stop"
    }
  ]
}
```

## Python SDK

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
    temperature=0.7,
    max_tokens=2048,
)

print(response.choices[0].message.content)
```

## 请求参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `model` | string | `local` | 兼容 OpenAI SDK 的占位模型名 |
| `messages` | array | 必填 | 原样传给 Ollama chat，复刻旧版 llama-cpp 行为 |
| `temperature` | number | `0.7` | 生成温度 |
| `max_tokens` | integer | `2048` | 最大输出 token 数 |
| `stream` | boolean | `false` | 当前不支持 SSE 流式返回 |

服务端不再额外切块、加 prompt 或改写客户端参数；`temperature` 和 `max_tokens` 会按请求原样传给模型。

空 `messages` 会返回：

```json
{"detail":"messages cannot be empty"}
```
