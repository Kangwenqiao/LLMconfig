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
qwen3-aigc-chat:latest
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

## 守护进程

当前使用 `scripts/watchdog.sh` 守护。它会检查并自动拉起：

- Ollama：`127.0.0.1:11434`
- API：`0.0.0.0:8000`
- nginx：`0.0.0.0:80`

启动守护循环：

```bash
cd /root/LLMconfig
export PATH="$HOME/.local/bin:$PATH"
PROJECT_DIR=/root/LLMconfig OLLAMA_MODEL=qwen3-aigc-chat:latest \
OLLAMA_KEEP_ALIVE=24h SERVER_PORT=8000 \
  scripts/watchdog.sh install
```

## 重启

```bash
ssh -p 23 root@117.50.89.11
cd /root/LLMconfig
export PATH="$HOME/.local/bin:$PATH"
scripts/watchdog.sh once
```

日志：

```text
/root/LLMconfig/watchdog.log
/root/LLMconfig/server.log
/root/ollama.log
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
  "ollama_model": "qwen3-aigc-chat:latest",
  "available_models": ["qwen3-aigc-chat:latest"]
}
```

## 性能配置

当前服务端按旧版 llama-cpp 语义运行：客户端 `messages`、`temperature`、`max_tokens` 原样传给 Ollama `/api/chat`。Ollama 模型必须使用显式 Qwen3 ChatML 模板，否则效果会退化。
