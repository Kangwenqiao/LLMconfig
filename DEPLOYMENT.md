# AIGC Rewriter 服务部署完成

## 服务信息

- **服务器地址**: `http://117.50.218.77`
- **API 格式**: OpenAI Compatible API
- **模型名称**: `local`

## API 使用示例

### 1. 健康检查

```bash
curl http://117.50.218.77/
```

响应：
```json
{"status":"ok","model":"qwen3-merged-aigc_zhv3-Q4_K_M","gpu_loaded":true,"workers":1}
```

### 2. 列出模型

```bash
curl http://117.50.218.77/v1/models
```

### 3. 聊天补全（OpenAI 兼容格式）

```bash
curl -X POST http://117.50.218.77/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [{"role": "user", "content": "你的文本内容"}],
    "temperature": 0.7,
    "max_tokens": 2048
  }'
```

### 4. Python SDK 使用

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://117.50.218.77/v1",
    api_key="not-needed"  # 本地服务不需要API key
)

response = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "人工智能技术的发展给人类社会带来了深远的影响。"}],
    temperature=0.7
)

print(response.choices[0].message.content)
```

## 服务管理

### 查看服务状态
```bash
ssh ubuntu@117.50.218.77 'sudo systemctl status aigc-rewriter'
```

### 重启服务
```bash
ssh ubuntu@117.50.218.77 'sudo systemctl restart aigc-rewriter'
```

### 查看日志
```bash
ssh ubuntu@117.50.218.77 'sudo journalctl -u aigc-rewriter -f'
```

## 部署架构

- **应用服务**: systemd service `aigc-rewriter.service`
- **反向代理**: nginx (监听 80 端口)
- **内部端口**: 8000
- **GPU**: NVIDIA RTX 2080 (8GB)
- **模型**: qwen3-merged-aigc_zhv3-Q4_K_M.gguf (~1.1GB)