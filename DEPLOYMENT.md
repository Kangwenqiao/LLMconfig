# 部署记录

## 当前线上服务

- 服务器：`117.50.89.11`
- SSH：`ssh -p 23 root@117.50.89.11`
- API：`http://117.50.89.11`
- OpenAI base_url：`http://117.50.89.11/v1`
- 后端监听：`0.0.0.0:8000`
- 反向代理：nginx，监听 `80` 并代理到 `127.0.0.1:8000`
- Ollama：`127.0.0.1:11434`

## 模型

只保留并使用一个 Ollama 模型：

```text
qwen3-aigc:latest
```

来源：

```text
https://huggingface.co/skskk/aigc-rewriter/resolve/main/qwen3-merged-aigc_zhv3-Q4_K_M.gguf
```

服务器文件：

```text
/root/LLMconfig/models/qwen3-merged-aigc_zhv3-Q4_K_M.gguf
```

模型文件 SHA256：

```text
24aff5a1cabe063800348fc1a8c0ef4cedfac338d1aa84fd7eaef1eb4d8c1734
```

## DNS

服务器已按云厂商要求临时写入：

```text
nameserver 100.90.90.90
nameserver 100.90.90.100
```

位置：

```text
/etc/resolv.conf
```

## 启动命令

当前使用 `nohup` 启动：

```bash
cd /root/LLMconfig
export PATH="$HOME/.local/bin:$PATH"
OLLAMA_MODEL=qwen3-aigc:latest SERVER_PORT=8000 \
  nohup uv run aigc_rewriter_server.py > server.log 2>&1 &
```

## 重启

```bash
ssh -p 23 root@117.50.89.11
cd /root/LLMconfig
PIDS=$(ss -ltnp 2>/dev/null | sed -n 's/.*:8000.*pid=\([0-9][0-9]*\).*/\1/p' | sort -u)
if [ -n "$PIDS" ]; then kill $PIDS; sleep 1; fi
export PATH="$HOME/.local/bin:$PATH"
OLLAMA_MODEL=qwen3-aigc:latest OLLAMA_KEEP_ALIVE=24h \
SERVER_MIN_TOKENS=128 SERVER_MAX_TOKENS=512 SERVER_MAX_TEMPERATURE=0.45 \
SENTENCES_PER_CALL=5 CHARS_PER_CALL=800 CHUNK_CONCURRENCY=2 SERVER_PORT=8000 \
  nohup uv run aigc_rewriter_server.py > server.log 2>&1 &
```

## 验证

```bash
curl http://117.50.89.11/
curl http://117.50.89.11/v1/models
curl -X POST http://117.50.89.11/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"local","messages":[{"role":"user","content":"人工智能技术的发展正在深刻改变人类社会的生产方式和生活方式。"}]}'
```

健康检查应只显示：

```json
{
  "ollama_model": "qwen3-aigc:latest",
  "available_models": ["qwen3-aigc:latest"]
}
```

## 性能配置

当前服务端已针对客户端配置做限流：

- 客户端传 `max_tokens=2048` 时，服务端会按输入长度自适应，默认最多生成 `512` token。
- 客户端传 `temperature=0.7` 时，服务端实际最高使用 `0.45`。
- 默认每 `5` 句送入一次模型，避免过度切碎影响降 AIGC 效果。
- Ollama 请求带 `keep_alive=24h`，`ollama ps` 应显示 `24 hours from now`。

短句单次本机 API 调用通常约 `1s`；长段会给更多输出空间，耗时随段落长度增加。如果客户端启用 `rewrite_rounds=3`，总耗时会接近单次耗时的 3 倍。
