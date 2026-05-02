# AIGC Rewriter API 文档

基于 `skskk/aigc-rewriter` GGUF 模型的文本重写服务

## 基本信息

- **服务地址**: `http://10.168.59.67:1051`
- **模型**: qwen3-merged-aigc_zhv3-Q4_K_M (Qwen3 架构)
- **上下文长度**: 4096 tokens
- **支持语言**: 中文

---

## 接口列表

### 1. 健康检查

检查服务运行状态。

**请求**

```
GET /
```

**响应示例**

```json
{
  "status": "ok",
  "model": "aigc-rewriter",
  "message": "服务运行中"
}
```

---

### 2. 文本重写（JSON格式）

重写输入文本，使其表达更加自然流畅。

**请求**

```
POST /rewrite
Content-Type: application/json
```

**请求体**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| text | string | 是 | - | 需要重写的文本 |
| temperature | float | 否 | 0.7 | 生成温度 (0.0-1.0)，值越高输出越随机 |
| max_tokens | int | 否 | 2048 | 最大生成token数 |

**请求示例**

```bash
curl -X POST "http://10.168.59.67:1051/rewrite" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "人工智能技术的发展给人类社会带来了深远的影响，它不仅改变了我们的生活方式，还推动了各行各业的变革。",
    "temperature": 0.7,
    "max_tokens": 2048
  }'
```

**响应**

| 字段 | 类型 | 说明 |
|------|------|------|
| rewritten_text | string | 重写后的文本 |
| original_length | int | 原文长度（字符数） |
| rewritten_length | int | 重写后长度（字符数） |

**响应示例**

```json
{
  "rewritten_text": "人工智能技术的发展给人类社会带来深远的影响，它改变着人们的日常生活，也推动着各个行业发生变革。",
  "original_length": 49,
  "rewritten_length": 47
}
```

---

### 3. 文本重写（简单接口）

通过URL参数进行简单的文本重写。

**请求**

```
GET /rewrite/simple?text=你的文本&temperature=0.7
```

或

```
POST /rewrite/simple?text=你的文本&temperature=0.7
```

**参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| text | string | 是 | - | 需要重写的文本（需URL编码） |
| temperature | float | 否 | 0.7 | 生成温度 (0.0-1.0) |

**请求示例**

```bash
# GET 请求
curl "http://10.168.59.67:1051/rewrite/simple?text=%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E6%8A%80%E6%9C%AF"

# POST 请求
curl -X POST "http://10.168.59.67:1051/rewrite/simple?text=%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E6%8A%80%E6%9C%AF"
```

**响应示例**

```json
{
  "result": "人工智能技术"
}
```

---

## 错误响应

当请求出错时，返回以下格式：

```json
{
  "detail": "错误描述"
}
```

**HTTP 状态码**

| 状态码 | 说明 |
|--------|------|
| 400 | 请求参数错误（如文本为空） |
| 500 | 服务器内部错误 |
| 503 | 模型未加载 |

---

## 启动与停止

**启动服务**

```bash
# 前台运行
uv run aigc_rewriter_server.py

# 后台运行
nohup uv run aigc_rewriter_server.py > server.log 2>&1 &
```

**停止服务**

```bash
# 查找进程
ps aux | grep aigc_rewriter_server

# 停止服务（优雅退出，自动释放GPU）
pkill -f aigc_rewriter_server
```

---

## Python 调用示例

```python
import requests

# POST 请求
response = requests.post(
    "http://10.168.59.67:1051/rewrite",
    json={
        "text": "人工智能技术正在改变世界",
        "temperature": 0.7
    }
)
result = response.json()
print(result["rewritten_text"])

# GET 请求
import urllib.parse
text = urllib.parse.quote("人工智能技术正在改变世界")
response = requests.get(f"http://10.168.59.67:1051/rewrite/simple?text={text}")
print(response.json()["result"])
```

---

## JavaScript 调用示例

```javascript
// POST 请求
const response = await fetch('http://10.168.59.67:1051/rewrite', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    text: '人工智能技术正在改变世界',
    temperature: 0.7
  })
});
const result = await response.json();
console.log(result.rewritten_text);

// GET 请求
const text = encodeURIComponent('人工智能技术正在改变世界');
const response = await fetch(`http://10.168.59.67:1051/rewrite/simple?text=${text}`);
const result = await response.json();
console.log(result.result);
```

---

## 模型信息

| 属性 | 值 |
|------|-----|
| 模型名称 | skskk/aigc-rewriter |
| 架构 | Qwen3 |
| 量化 | Q4_K_M |
| 模型大小 | ~1.1 GB |
| 上下文长度 | 40960 (训练) / 4096 (服务) |
| 缓存路径 | `~/.cache/aigc-rewriter/` |

---

## 注意事项

1. 服务启动时需要加载模型到GPU，首次启动约需10-20秒
2. 停止服务时会自动释放GPU资源
3. 使用 `--cpu` 参数可切换为纯CPU模式（暂未在服务中实现）
4. 建议使用POST接口处理长文本，GET接口有URL长度限制
