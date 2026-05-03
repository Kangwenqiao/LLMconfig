# AIGC Rewriter API 文档

基于 `skskk/aigc-rewriter` GGUF 模型的文本重写服务

## 基本信息

- **服务地址**: `http://10.168.59.67:1051`
- **模型**: qwen3-merged-aigc_zhv3-Q4_K_M (Qwen3 架构)
- **上下文长度**: 4096 tokens
- **支持语言**: 中文
- **GPU模式**: 默认 `AIGC_REWRITER_GPU_LAYERS=-1`，模型层尽量全部加载到 GPU
- **并行能力**: 默认 `AIGC_REWRITER_WORKERS=1`；需要真实并行时可提高 worker 数，每个 worker 会加载一个独立模型实例
- **默认降重强度**: `high`
- **提示词模式**: 默认 `minimal`，不发送 system prompt、不发送 JSON schema，只在 user 消息中加入一行极短约束，要求保留占位符、不扩写、只输出正文

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
  "message": "服务运行中",
  "workers": 1,
  "n_gpu_layers": -1,
  "n_ctx": 4096,
  "gpu_loaded": false,
  "idle_timeout": 300,
  "default_strength": "high"
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
| strength | string | 否 | high | 降重强度：`low`、`medium`、`high` |

**请求示例**

```bash
curl -X POST "http://10.168.59.67:1051/rewrite" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "人工智能技术的发展给人类社会带来了深远的影响，它不仅改变了我们的生活方式，还推动了各行各业的变革。",
    "temperature": 0.7,
    "max_tokens": 2048,
    "strength": "high"
  }'
```

**响应**

| 字段 | 类型 | 说明 |
|------|------|------|
| rewritten_text | string | 重写后的文本 |
| original_length | int | 原文长度（字符数） |
| rewritten_length | int | 重写后长度（字符数） |
| strength | string | 实际使用的降重强度 |
| similarity | float | 与原文的字面相似度估算，0-1之间，越低表示字面重复越少 |
| passes | int | 实际改写轮数；`high` 模式相似度偏高时最多自动精修2轮 |

**响应示例**

```json
{
  "rewritten_text": "人工智能技术的发展给人类社会带来深远的影响，它改变着人们的日常生活，也推动着各个行业发生变革。",
  "original_length": 49,
  "rewritten_length": 47,
  "strength": "high",
  "similarity": 0.48,
  "passes": 2
}
```

---

### 3. 批量文本重写（支持并行）

批量重写多段文本。结果顺序与输入 `texts` 顺序一致。

实际并行度受服务启动时的 `AIGC_REWRITER_WORKERS` 限制。例如 `AIGC_REWRITER_WORKERS=2` 时，最多同时执行 2 个模型推理任务。

**请求**

```
POST /rewrite/batch
Content-Type: application/json
```

**请求体**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| texts | string[] | 是 | - | 需要重写的文本列表 |
| temperature | float | 否 | 0.7 | 生成温度 (0.0-1.0) |
| max_tokens | int | 否 | 2048 | 每条文本最大生成token数 |
| strength | string | 否 | high | 降重强度：`low`、`medium`、`high` |
| concurrency | int | 否 | workers数量 | 本次批量请求的并行度，上限为服务 workers 数 |

**请求示例**

```bash
curl -X POST "http://10.168.59.67:1051/rewrite/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "人工智能技术正在改变人们的生活方式。",
      "大数据技术为企业决策提供了重要支持。"
    ],
    "temperature": 0.7,
    "max_tokens": 2048,
    "strength": "high",
    "concurrency": 2
  }'
```

**响应**

| 字段 | 类型 | 说明 |
|------|------|------|
| results | object[] | 每条文本的重写结果 |
| count | int | 返回结果数量 |
| workers | int | 当前服务加载的模型实例数 |

`results` 内字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| index | int | 输入文本下标 |
| rewritten_text | string | 重写后的文本 |
| original_length | int | 原文长度（字符数） |
| rewritten_length | int | 重写后长度（字符数） |
| strength | string | 实际使用的降重强度 |
| similarity | float | 与原文的字面相似度估算 |
| passes | int | 实际改写轮数 |

**响应示例**

```json
{
  "results": [
    {
      "index": 0,
      "rewritten_text": "人工智能技术的发展正在改变人们的生活方式。",
      "original_length": 19,
      "rewritten_length": 22,
      "strength": "high",
      "similarity": 0.51,
      "passes": 2
    },
    {
      "index": 1,
      "rewritten_text": "大数据技术为企业制定决策提供了重要依据。",
      "original_length": 20,
      "rewritten_length": 22,
      "strength": "high",
      "similarity": 0.46,
      "passes": 1
    }
  ],
  "count": 2,
  "workers": 2
}
```

---

### 4. 文本重写（简单接口）

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
  "result": "人工智能技术",
  "strength": "high",
  "similarity": 1.0,
  "passes": 1
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

**并行启动示例**

```bash
# 默认使用GPU，容器内监听1001；外部通过1051映射访问
AIGC_REWRITER_WORKERS=2 nohup uv run aigc_rewriter_server.py > server.log 2>&1 &
```

并行相关环境变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| AIGC_REWRITER_WORKERS | 1 | 模型实例数；真实并行度上限。每增加1个worker都会额外占用显存和内存 |
| AIGC_REWRITER_GPU_LAYERS | -1 | GPU加载层数；-1表示尽量全部加载到GPU |
| AIGC_REWRITER_N_CTX | 4096 | 服务上下文长度 |
| AIGC_REWRITER_MAX_BATCH_SIZE | 32 | `/rewrite/batch` 单次最多文本数量 |
| AIGC_REWRITER_DEFAULT_STRENGTH | high | 默认降重强度；可选 `low`、`medium`、`high` |
| AIGC_REWRITER_IDLE_TIMEOUT | 300 | 空闲多少秒后释放模型资源 |

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
        "temperature": 0.7,
        "strength": "high"
    }
)
result = response.json()
print(result["rewritten_text"])

# GET 请求
import urllib.parse
text = urllib.parse.quote("人工智能技术正在改变世界")
response = requests.get(f"http://10.168.59.67:1051/rewrite/simple?text={text}")
print(response.json()["result"])

# 批量并行请求
response = requests.post(
    "http://10.168.59.67:1051/rewrite/batch",
    json={
        "texts": [
            "人工智能技术正在改变世界",
            "大数据技术正在推动产业升级"
        ],
        "temperature": 0.7,
        "strength": "high",
        "concurrency": 2
    }
)
for item in response.json()["results"]:
    print(item["index"], item["rewritten_text"])
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
    temperature: 0.7,
    strength: 'high'
  })
});
const result = await response.json();
console.log(result.rewritten_text);

// GET 请求
const text = encodeURIComponent('人工智能技术正在改变世界');
const response = await fetch(`http://10.168.59.67:1051/rewrite/simple?text=${text}`);
const result = await response.json();
console.log(result.result);

// 批量并行请求
const batchResponse = await fetch('http://10.168.59.67:1051/rewrite/batch', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    texts: [
      '人工智能技术正在改变世界',
      '大数据技术正在推动产业升级'
    ],
    temperature: 0.7,
    strength: 'high',
    concurrency: 2
  })
});
const batchResult = await batchResponse.json();
console.log(batchResult.results);
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
| 模型路径 | `models/qwen3-merged-aigc_zhv3-Q4_K_M.gguf` |

---

## 注意事项

1. 服务启动时需要加载模型到GPU，首次启动约需10-20秒
2. 停止服务时会自动释放GPU资源
3. 服务默认使用 GPU：`AIGC_REWRITER_GPU_LAYERS=-1`
4. 提高 `AIGC_REWRITER_WORKERS` 才能获得真实模型推理并行；并发数越高，显存和内存占用越高
5. 单 worker 时多个请求会排队执行，但健康检查和连接处理不会被模型推理直接阻塞
6. 建议客户端批量降重时使用 `/rewrite/batch`，并将 `concurrency` 控制在服务 `workers` 以内
7. 建议使用POST接口处理长文本，GET接口有URL长度限制
8. 当前默认 `AIGC_REWRITER_PROMPT_MODE=minimal`，服务不发送 system prompt、不使用 JSON schema；只发送极短 user 约束，要求原样保留 `@@...@@` 占位符、数字、术语和引用标记，不新增内容、不扩写；该模式只执行1轮模型输出
9. 服务会拦截提示词泄露内容，例如“必须进行深度改写”“改变句式结构”“rewritten_text”等规则文本，不会把这类内容作为正文直接返回；如果输入本身就是这类残留段落，会返回空正文，便于上游删除该段
10. `raw` 模式用于裸输入测试，容易丢失引用占位符或自由扩写，不建议正式整篇处理；`minimal` 模式用于正式重跑
11. 服务会跳过参考文献、图题、表题、章节标题、关键词等结构化文本，避免破坏格式
