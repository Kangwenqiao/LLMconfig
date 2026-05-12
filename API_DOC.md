# AIGC Rewriter API 文档

## 基本信息

- 服务地址：`http://117.50.89.11`
- OpenAI base_url：`http://117.50.89.11/v1`
- 客户端 model：`local`
- Ollama 模型：`qwen3-aigc:latest`
- GGUF 模型：`qwen3-merged-aigc_zhv3-Q4_K_M.gguf`
- 模型来源：`skskk/aigc-rewriter`
- 推理方式：Ollama `/api/generate`，`raw: true`

## 健康检查

```bash
curl http://117.50.89.11/
```

响应示例：

```json
{
  "status": "ok",
  "ollama_host": "http://localhost:11434",
  "ollama_model": "qwen3-aigc:latest",
  "available_models": ["qwen3-aigc:latest"]
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
      "id": "qwen3-aigc:latest",
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
| `messages` | array | 必填 | 取最后一条 `user` 消息作为待改写文本 |
| `temperature` | number | `0.2` | 生成温度 |
| `max_tokens` | integer | `64` | 最大输出 token 数 |
| `stream` | boolean | `false` | 当前不支持 SSE 流式返回 |

服务端会兼容客户端较大的配置，但会做硬限制：

- `max_tokens` 会按输入长度自适应，默认范围是 `64-256`；即使客户端传 `2048` 也不会照单生成 2048 token。
- `temperature` 上限默认是 `0.3`，即使客户端传 `0.7` 也会压低，减少复读和跑题。
- Ollama `keep_alive` 默认是 `24h`，模型会尽量保持 GPU 常驻。

空 `messages` 会返回：

```json
{"detail":"messages cannot be empty"}
```
