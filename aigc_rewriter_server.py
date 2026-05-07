#!/usr/bin/env python3
"""
AIGC Rewriter API Service
基于 skskk/aigc-rewriter GGUF 模型的文本重写服务
"""

import os
import gc
import re
import json
import asyncio
import time
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama
import uvicorn

# 模型配置 - 下载到项目路径的models目录
MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "qwen3-merged-aigc_zhv3-Q4_K_M.gguf"
MODEL_WORKERS = max(1, int(os.environ.get("AIGC_REWRITER_WORKERS", "2")))
N_GPU_LAYERS = int(os.environ.get("AIGC_REWRITER_GPU_LAYERS", "-1"))
N_CTX = int(os.environ.get("AIGC_REWRITER_N_CTX", "4096"))
MAX_BATCH_SIZE = max(1, int(os.environ.get("AIGC_REWRITER_MAX_BATCH_SIZE", "32")))
DEFAULT_STRENGTH = os.environ.get("AIGC_REWRITER_DEFAULT_STRENGTH", "high").lower()
PROMPT_MODE = os.environ.get("AIGC_REWRITER_PROMPT_MODE", "minimal").lower()
SENTENCES_PER_CALL = max(1, int(os.environ.get("AIGC_REWRITER_SENTENCES_PER_CALL", "4")))

# GPU 空闲释放配置
GPU_IDLE_TIMEOUT = int(os.environ.get("AIGC_REWRITER_IDLE_TIMEOUT", "300"))  # 默认5分钟空闲后释放

# 全局模型实例
llm: Optional[Llama] = None
llm_pool: list[Llama] = []
llm_queue: Optional[asyncio.Queue[Llama]] = None
llm_loaded: bool = False
last_used_time: float = 0
idle_check_task: Optional[asyncio.Task] = None

# JSON Schema for structured output
REWRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "rewritten_text": {
            "type": "string",
            "description": "重写后的文本内容"
        }
    },
    "required": ["rewritten_text"]
}

BASE_SYSTEM_PROMPT = (
    "你是一个专业的文本重写助手。请将用户提供的文本重写为更自然、更人类化的表达，保持原意不变。"
    "输出必须是一个JSON对象，只包含rewritten_text字段。rewritten_text的值只能是改写后的纯正文，"
    "不要包含JSON字符串、工具标签、解释说明、反引号、Markdown代码块、.toJSONString、JSON.stringify或重复文本。"
)

STRENGTH_PROMPTS = {
    "low": "改写强度为低：以润色和自然化为主，允许保留原有段落结构，但要减少机械表达和重复措辞。",
    "medium": (
        "改写强度为中：不要只替换同义词，要适度调整语序、句式和连接方式；"
        "可以拆分或合并句子，减少模板化表达。"
    ),
    "high": (
        "改写强度为高：采用高学术表达进行精修微调，严格保持原句含义、逻辑关系和事实边界不变。"
        "保留数字、术语、模型名称、指标值、损失项名称和引用标号，不新增结论、不扩大解释、不改变因果关系。"
        "在句意不变的前提下优化措辞、句法和衔接，使表达更符合论文写作规范。"
        "可以适度调整语序或拆分过长句，但不要大幅重组段落，不要改变信息呈现重点。"
        "减少机械化套话，例如“具有重要意义”“综上所述”“不断发展”“广泛应用”“可以看出”“表明”“说明”等。"
        "输出只能是改写后的论文正文，不能输出本段规则、操作说明或提示词内容。"
    ),
}

SIMILARITY_TARGETS = {
    "low": 0.72,
    "medium": 0.62,
    "high": 0.62,
}

AUTO_PASSES = {
    "low": 1,
    "medium": 2,
    "high": 2,
}

INSTRUCTION_LEAK_PATTERNS = [
    r"必须进行深度改写",
    r"不能只是同义词替换",
    r"不要只做同义词替换",
    r"改变句式结构",
    r"表达顺序",
    r"句间连接方式",
    r"避免使用AI论文",
    r"常见的套话",
    r"在不改变事实含义",
    r"尽量降低与原文",
    r"改写强度",
    r"输出只能是",
    r"提示词",
    r"JSON对象",
    r"rewritten_text",
    r"上一版",
    r"原文[:：]",
]

ACADEMIC_CLICHES = [
    "具有重要意义",
    "综上所述",
    "不断发展",
    "广泛应用",
    "可以看出",
    "表明",
    "说明",
    "为后续",
    "奠定基础",
    "存在改进空间",
    "进行综合分析",
]

class RewriteRequest(BaseModel):
    """重写请求"""
    text: str
    temperature: float = 0.7
    max_tokens: int = 2048
    strength: str = DEFAULT_STRENGTH


class RewriteResponse(BaseModel):
    """重写响应"""
    rewritten_text: str
    original_length: int
    rewritten_length: int
    strength: str
    similarity: float
    passes: int


class BatchRewriteRequest(BaseModel):
    """批量重写请求"""
    texts: list[str]
    temperature: float = 0.7
    max_tokens: int = 2048
    strength: str = DEFAULT_STRENGTH
    concurrency: Optional[int] = None


class BatchRewriteItem(BaseModel):
    """批量重写单条响应"""
    index: int
    rewritten_text: str
    original_length: int
    rewritten_length: int
    strength: str
    similarity: float
    passes: int


class BatchRewriteResponse(BaseModel):
    """批量重写响应"""
    results: list[BatchRewriteItem]
    count: int
    workers: int


def _remove_model_markup(text: str) -> str:
    """移除模型可能输出的思考、代码块和工具标签。"""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<\|[^>]+?\|>", "", text)
    text = re.sub(
        r"</?(?:tool_call|tool_response|tool|function_call|function_response)[^>]*>",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip()


def _truncate_artifact_suffix(text: str) -> str:
    """截断正文后面混入的代码、工具或序列化残留。"""
    artifact_patterns = [
        r"<(?:tool_call|tool_response|tool|function_call|function_response)\b",
        r"\.toJSONString\b",
        r"\btoJSONString\s*\(",
        r"\bJSON\.stringify\b",
        r"\bjson\.dumps\b",
        r"```",
        r"(?<![\w\u4e00-\u9fff])(?:原文|上一版|当前改写版本|改写强度|要求)[:：]?",
    ]
    for pattern in artifact_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            if match.start() == 0:
                text = text[match.end():]
            else:
                text = text[:match.start()]
    return text.strip()


def _extract_json_rewritten_text(text: str) -> Optional[str]:
    """从完整或嵌入式 JSON 中提取 rewritten_text。"""
    decoder = json.JSONDecoder()
    candidates = [text.strip()]

    for match in re.finditer(r"\{", text):
        candidates.append(text[match.start():].strip())

    for candidate in candidates:
        try:
            data, _ = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and isinstance(data.get("rewritten_text"), str):
            return data["rewritten_text"]

    match = re.search(r'"rewritten_text"\s*:\s*("(?:[^"\\]|\\.)*")', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return match.group(1).strip('"')

    return None


def _clean_json_residue(text: str) -> str:
    """移除夹在正文标点后的 JSON 闭合残片。"""
    text = re.sub(r"([。！？!?\.]['\"’”]?)\s*[}\]]+(?=\s|$)", r"\1", text)
    text = re.sub(r"([。！？!?\.]['\"’”]?)\s*[}\]]+(?=[^\w\u4e00-\u9fff]|$)", r"\1", text)
    text = re.sub(r"([。！？!?\.]['\"’”]?)\s*[}\]]+", r"\1", text)
    text = re.sub(r"\s*[,，]\s*[}\]]+\s*$", "", text)
    return text.strip()


def _dedupe_repeated_text(text: str) -> str:
    """清理模型偶发的整段重复或连续重复句。"""
    normalized = lambda value: re.sub(r"\s+", "", value)

    if len(text) >= 20:
        for unit_len in range(8, len(text) // 2 + 1):
            unit = text[:unit_len].strip()
            if not unit:
                continue
            repeated = unit
            count = 1
            while len(repeated) < len(text) and text.startswith(repeated + unit):
                repeated += unit
                count += 1
            if count >= 2 and normalized(text[len(repeated):]) in {"", normalized(unit[: len(text) - len(repeated)])}:
                text = unit
                break

        middle = len(text) // 2
        window = min(80, middle)
        for split_at in range(max(1, middle - window), min(len(text), middle + window) + 1):
            left = text[:split_at].strip()
            right = text[split_at:].strip()
            if left and normalized(left) == normalized(right):
                text = left
                break

    parts = re.findall(r".+?[。！？!?](?:[’”])?|.+$", text, flags=re.DOTALL)
    cleaned_parts: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if cleaned_parts and normalized(cleaned_parts[-1]) == normalized(part):
            continue
        cleaned_parts.append(part)

    return "".join(cleaned_parts).strip()


def _normalize_technical_terms(text: str) -> str:
    """修正常见技术术语被模型改坏的大小写。"""
    text = re.sub(r"(?i)yolo(?=[\u4e00-\u9fffv\d-])", "YOLO", text)
    text = re.sub(r"(?i)\bnms-free\b", "NMS-free", text)
    text = re.sub(r"(?i)\bsgd\b", "SGD", text)
    text = re.sub(r"(?i)\bgflops\b", "GFLOPs", text)
    text = re.sub(r"(?i)\bmap50-95\b", "mAP50-95", text)
    text = re.sub(r"(?i)\bmap50\b", "mAP50", text)
    return text


def _missing_placeholders(original_text: str, rewritten_text: str) -> list[str]:
    """返回模型输出中丢失的上游保护占位符。"""
    placeholders = re.findall(r"@@[A-Z_]+_\d+@@", original_text)
    return [placeholder for placeholder in placeholders if placeholder not in rewritten_text]


def _split_sentences(text: str) -> list[str]:
    """按中文论文常见句末标点切分，保留句末标点和紧随其后的引用占位符。"""
    pattern = r".+?(?:[。！？!?](?:@@[A-Z_]+_\d+@@)*|$)"
    sentences = [match.group(0).strip() for match in re.finditer(pattern, text, flags=re.DOTALL)]
    return [sentence for sentence in sentences if sentence]


def _group_sentences(sentences: list[str], size: int) -> list[str]:
    return ["".join(sentences[index:index + size]) for index in range(0, len(sentences), size)]


def extract_pure_text(content: str, original_text: str) -> str:
    """
    从模型输出中提取纯净的重写文本。

    API 的 rewritten_text 必须只包含正文；这里兜底清理完整 JSON、半截 JSON、
    工具标签、代码块、解释说明和偶发重复输出。
    """
    text = _remove_model_markup(content or "")

    extracted = _extract_json_rewritten_text(text)
    if extracted is not None:
        text = extracted
        nested = _extract_json_rewritten_text(_remove_model_markup(text))
        if nested is not None:
            text = nested

    text = _truncate_artifact_suffix(text)
    text = _remove_model_markup(text)
    text = re.split(
        r"<(?:tool_call|tool_response|tool|function_call|function_response)\b",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]

    text = re.sub(
        r"^\s*\{?\s*['\"]?rewritten_text['\"]?\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = text.strip().strip("` \t\r\n")

    if text.startswith('"') and text.endswith('"'):
        try:
            text = json.loads(text)
        except json.JSONDecodeError:
            text = text[1:-1]

    text = _truncate_artifact_suffix(text)
    text = _clean_json_residue(text)
    text = re.sub(r"([。！？!?\.])['\"’”]?\s*[}\]\{]+[\s}\]\{'\"`]*$", r"\1", text)
    text = re.sub(r"^[\s{\[}\]'\"`]+", "", text)
    text = re.sub(r"[\s{\[}\]'\"`]+$", "", text)

    text = _dedupe_repeated_text(_clean_json_residue(text.strip()))
    text = _normalize_technical_terms(text)
    return text.strip()


def normalize_strength(strength: str) -> str:
    """规范化改写强度。"""
    value = (strength or DEFAULT_STRENGTH).lower().strip()
    if value not in STRENGTH_PROMPTS:
        raise ValueError("strength必须是low、medium或high")
    return value


def build_system_prompt(strength: str) -> str:
    """按强度构造系统提示词。"""
    return f"{BASE_SYSTEM_PROMPT}{STRENGTH_PROMPTS[strength]}"


def output_quality_penalty(text: str, strength: str) -> float:
    """估计输出中的提示词泄露和论文套话风险。"""
    if not text.strip():
        return 1.0

    penalty = 0.0
    for pattern in INSTRUCTION_LEAK_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            penalty += 1.0

    if strength == "high":
        penalty += sum(0.05 for phrase in ACADEMIC_CLICHES if phrase in text)

    return round(penalty, 4)


def is_valid_rewrite(text: str, original_text: str, strength: str) -> bool:
    """判断候选结果是否可作为正文返回。"""
    if not text.strip():
        return False
    if output_quality_penalty(text, strength) >= 1.0:
        return False
    # 极短文本不做长度约束；长文本若缩水过度，通常是模型输出说明或截断。
    if len(original_text) >= 80 and len(text) < len(original_text) * 0.45:
        return False
    return True


def is_instruction_residue(text: str) -> bool:
    """判断输入本身是否是误写入正文的系统提示残留。"""
    return output_quality_penalty(text, "high") >= 1.0


def should_skip_rewrite(text: str) -> bool:
    """跳过参考文献、图表标题、章节标题等不适合改写的结构化文本。"""
    stripped = text.strip()
    if not stripped:
        return True

    patterns = [
        r"^\[\d+\]\s+.+(?:DOI:|doi:|https?://|\[J\]|\[EB/OL\])",
        r"^图\s*\d+(?:-\d+)?\s+.+",
        r"^表\s*\d+(?:-\d+)?\s+.+",
        r"^\d+(?:\.\d+){0,3}\s+\S.{0,30}$",
        r"^(关键词|Key Words|Key words)\s*[:：]",
    ]
    if any(re.search(pattern, stripped) for pattern in patterns):
        return True
    if len(stripped) <= 24 and re.search(r"(研究|实验|分析|比较|设置|展望|模型|数据集|YOLO|Angular|Turning|Strawberry)", stripped):
        return True
    return False


def _similarity_text(text: str) -> str:
    """用于相似度计算的轻量规范化文本。"""
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", text.lower())


def _char_ngrams(text: str) -> set[str]:
    normalized = _similarity_text(text)
    if not normalized:
        return set()
    size = 3 if len(normalized) >= 12 else 2 if len(normalized) >= 4 else 1
    return {normalized[index:index + size] for index in range(max(1, len(normalized) - size + 1))}


def similarity_score(original_text: str, rewritten_text: str) -> float:
    """返回0-1之间的字面相似度，越高说明和原文越接近。"""
    original = _char_ngrams(original_text)
    rewritten = _char_ngrams(rewritten_text)
    if not original or not rewritten:
        return 0.0
    overlap = len(original & rewritten)
    jaccard = overlap / len(original | rewritten)
    containment = overlap / min(len(original), len(rewritten))
    return round(max(jaccard, containment * 0.85), 4)


def _build_user_content(text: str, original_text: str, pass_index: int, strength: str) -> str:
    if pass_index <= 1:
        return (
            "请改写下面的论文正文。只返回改写后的正文，不要解释，不要复述规则。\n\n"
            f"{text}"
        )

    if strength == "high":
        return (
            f"第{pass_index}轮精修：上一版表达仍需更学术、更自然，但必须保持句意不变。\n"
            "要求：\n"
            "1. 保留原文全部事实、数字、模型名称、技术名词、损失项名称和引用标号。\n"
            "2. 不改变原文的因果关系、比较关系、结论强弱和段落重点。\n"
            "3. 只做学术化精修、措辞优化和轻微句法调整，避免大幅重组。\n"
            "4. 去除机械套话和重复连接词，使句子更像人工论文表述。\n"
            "5. 只返回改写后的论文正文，不要输出规则说明。\n\n"
            f"原始文本：{original_text}\n\n"
            f"上一版改写：{text}"
        )

    return (
        "上一版改写仍然不够自然，或与原文过于接近。请继续改写正文。\n"
        "保留原始文本中的数字、模型名称、技术名词和引用标号；不要输出任何规则说明。\n\n"
        f"原始文本：{original_text}\n\n"
        f"当前改写版本：{text}\n\n"
        f"改写强度：{strength}"
    )


def _build_minimal_user_content(text: str) -> str:
    """极短提示词：不使用 system prompt 和 JSON schema，只约束保留占位符与不扩写。"""
    return (
        "改写下面的论文正文，保持原意，不新增内容，不扩写。"
        "必须原样保留所有@@...@@占位符、数字、术语和引用标记。"
        "长度接近原文。只输出改写后的正文。\n"
        f"{text}"
    )


def _create_rewrite_once(
    model: Llama,
    text: str,
    original_text: str,
    temperature: float,
    max_tokens: int,
    strength: str,
    pass_index: int,
) -> str:
    """在线程中执行同步模型推理。"""
    if PROMPT_MODE == "raw":
        messages = [{"role": "user", "content": text}]
        response = model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|im_end|>"],
        )
    elif PROMPT_MODE == "minimal":
        messages = [{"role": "user", "content": _build_minimal_user_content(text)}]
        response = model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|im_end|>"],
        )
    else:
        messages = [
            {"role": "system", "content": build_system_prompt(strength)},
            {"role": "user", "content": _build_user_content(text, original_text, pass_index, strength)}
        ]
        response = model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|im_end|>"],
            response_format={
                "type": "json_object",
                "schema": REWRITE_SCHEMA
            }
        )

    content = response["choices"][0]["message"]["content"]
    rewritten = extract_pure_text(content, original_text)
    if PROMPT_MODE in {"raw", "minimal"} and _missing_placeholders(original_text, rewritten):
        return original_text
    if PROMPT_MODE in {"raw", "minimal"} and len(original_text) >= 80 and len(rewritten) > len(original_text) * 1.35:
        return original_text
    return rewritten


def _create_rewrite(model: Llama, text: str, temperature: float, max_tokens: int, strength: str) -> tuple[str, float, int]:
    """按强度执行一次或多次同步模型推理。"""
    strength = normalize_strength(strength)
    original_text = text
    if should_skip_rewrite(original_text):
        return original_text, 1.0, 0
    if is_instruction_residue(original_text):
        return "", 0.0, 0

    sentences = _split_sentences(original_text)
    if SENTENCES_PER_CALL > 0 and len(sentences) > SENTENCES_PER_CALL:
        rewritten_chunks: list[str] = []
        calls = 0
        for chunk in _group_sentences(sentences, SENTENCES_PER_CALL):
            chunk_rewritten, _chunk_similarity, chunk_passes = _create_rewrite(
                model=model,
                text=chunk,
                temperature=temperature,
                max_tokens=max(256, min(max_tokens, 1024)),
                strength=strength,
            )
            calls += max(chunk_passes, 1)
            rewritten_chunks.append(chunk_rewritten)

        combined_text = "".join(rewritten_chunks)
        if _missing_placeholders(original_text, combined_text):
            return original_text, 1.0, calls
        if len(original_text) >= 80 and len(combined_text) > len(original_text) * 1.35:
            return original_text, 1.0, calls
        if len(original_text) >= 80 and len(combined_text) < len(original_text) * 0.45:
            return original_text, 1.0, calls
        return combined_text, similarity_score(original_text, combined_text), calls

    current_text = text
    best_text = text
    best_similarity = 1.0
    best_rank = float("inf")
    pass_count = 0
    target = SIMILARITY_TARGETS[strength]
    effective_temperature = max(temperature, 0.72) if strength == "high" else temperature

    max_passes = 1 if PROMPT_MODE in {"raw", "minimal"} else AUTO_PASSES[strength]

    for pass_index in range(1, max_passes + 1):
        pass_count = pass_index
        rewritten = _create_rewrite_once(
            model=model,
            text=current_text,
            original_text=original_text,
            temperature=effective_temperature,
            max_tokens=max_tokens,
            strength=strength,
            pass_index=pass_index,
        )
        score = similarity_score(original_text, rewritten)
        penalty = output_quality_penalty(rewritten, strength)
        rank = score + penalty
        if rewritten and penalty < 1.0 and (rank < best_rank or best_text == original_text):
            best_text = rewritten
            best_similarity = score
            best_rank = rank
        if score <= target and is_valid_rewrite(rewritten, original_text, strength):
            break

        if is_valid_rewrite(rewritten, original_text, strength):
            current_text = rewritten
        else:
            current_text = original_text

    return best_text, best_similarity, pass_count


async def _load_models() -> bool:
    """加载模型到GPU"""
    global llm, llm_pool, llm_queue, llm_loaded, last_used_time

    if llm_loaded:
        return True

    print(f"正在加载模型... workers={MODEL_WORKERS}, n_gpu_layers={N_GPU_LAYERS}, n_ctx={N_CTX}")
    llm_pool = []
    llm_queue = asyncio.Queue(maxsize=MODEL_WORKERS)

    for index in range(MODEL_WORKERS):
        model = Llama(
            model_path=str(MODEL_PATH),
            n_gpu_layers=N_GPU_LAYERS,
            n_ctx=N_CTX,
            verbose=False
        )
        llm_pool.append(model)
        llm_queue.put_nowait(model)
        if index == 0:
            llm = model

    llm_loaded = True
    last_used_time = time.time()
    print(f"模型加载完成，workers={len(llm_pool)}")
    return True


async def _release_models():
    """释放GPU资源"""
    global llm, llm_pool, llm_queue, llm_loaded

    if not llm_loaded:
        return

    print("正在释放GPU资源...")
    for model in llm_pool:
        del model
    llm_pool = []
    llm_queue = None
    llm = None
    llm_loaded = False
    gc.collect()
    print("GPU资源已释放")


async def _idle_checker():
    """后台任务：检查空闲时间，超时释放GPU"""
    while True:
        await asyncio.sleep(60)  # 每60秒检查一次
        if llm_loaded and last_used_time > 0:
            idle_time = time.time() - last_used_time
            if idle_time > GPU_IDLE_TIMEOUT:
                print(f"空闲 {idle_time:.0f} 秒，超过阈值 {GPU_IDLE_TIMEOUT} 秒，释放GPU")
                await _release_models()


async def run_rewrite(text: str, temperature: float, max_tokens: int, strength: str) -> tuple[str, float, int]:
    """从模型池取一个空闲实例执行重写；多实例时可并行。"""
    global last_used_time

    strength = normalize_strength(strength)
    if should_skip_rewrite(text):
        return text, 1.0, 0
    if is_instruction_residue(text):
        return "", 0.0, 0

    # 如果模型未加载，先加载
    if not llm_loaded:
        await _load_models()

    if llm_queue is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    last_used_time = time.time()  # 更新最后使用时间

    model = await llm_queue.get()
    try:
        return await asyncio.to_thread(_create_rewrite, model, text, temperature, max_tokens, strength)
    finally:
        llm_queue.put_nowait(model)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global idle_check_task

    # 启动时检查模型文件是否存在
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"模型文件不存在: {MODEL_PATH}\n"
            f"请先下载模型: proxychains4 wget -O {MODEL_PATH} "
            "https://huggingface.co/skskk/aigc-rewriter/resolve/main/qwen3-merged-aigc_zhv3-Q4_K_M.gguf"
        )

    print(f"服务已启动，GPU空闲超时: {GPU_IDLE_TIMEOUT}秒")
    print(f"模型将在首次调用时加载，workers={MODEL_WORKERS}, n_gpu_layers={N_GPU_LAYERS}")

    # 启动空闲检查任务
    idle_check_task = asyncio.create_task(_idle_checker())

    yield

    # 关闭时释放资源
    if idle_check_task:
        idle_check_task.cancel()
    await _release_models()


app = FastAPI(
    title="AIGC Rewriter API",
    description="降AIGC重写服务 - 将AI生成的内容改写为更加自然、人类化的表达",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """健康检查"""
    return {
        "status": "ok",
        "model": "aigc-rewriter",
        "message": "服务运行中",
        "workers": MODEL_WORKERS,
        "n_gpu_layers": N_GPU_LAYERS,
        "n_ctx": N_CTX,
        "gpu_loaded": llm_loaded,
        "idle_timeout": GPU_IDLE_TIMEOUT,
        "default_strength": DEFAULT_STRENGTH,
        "prompt_mode": PROMPT_MODE,
        "sentences_per_call": SENTENCES_PER_CALL,
    }


@app.post("/rewrite", response_model=RewriteResponse)
async def rewrite_text(request: RewriteRequest):
    """
    重写文本接口

    将输入的文本进行重写，使表达更加自然流畅
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="文本不能为空")

    try:
        strength = normalize_strength(request.strength)
        rewritten, similarity, passes = await run_rewrite(
            request.text,
            request.temperature,
            request.max_tokens,
            strength,
        )

        return RewriteResponse(
            rewritten_text=rewritten,
            original_length=len(request.text),
            rewritten_length=len(rewritten),
            strength=strength,
            similarity=similarity,
            passes=passes,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重写失败: {str(e)}")


@app.post("/rewrite/batch", response_model=BatchRewriteResponse)
async def rewrite_batch(request: BatchRewriteRequest):
    """
    批量重写接口

    多模型实例时可并行处理；结果顺序与输入 texts 顺序一致。
    """
    if not request.texts:
        raise HTTPException(status_code=400, detail="texts不能为空")
    if len(request.texts) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=400, detail=f"texts数量不能超过{MAX_BATCH_SIZE}")
    if any(not text.strip() for text in request.texts):
        raise HTTPException(status_code=400, detail="texts中不能包含空文本")

    try:
        strength = normalize_strength(request.strength)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    requested_concurrency = request.concurrency or MODEL_WORKERS
    concurrency = max(1, min(requested_concurrency, MODEL_WORKERS))
    semaphore = asyncio.Semaphore(concurrency)

    async def rewrite_one(index: int, text: str) -> BatchRewriteItem:
        async with semaphore:
            rewritten, similarity, passes = await run_rewrite(
                text,
                request.temperature,
                request.max_tokens,
                strength,
            )
            return BatchRewriteItem(
                index=index,
                rewritten_text=rewritten,
                original_length=len(text),
                rewritten_length=len(rewritten),
                strength=strength,
                similarity=similarity,
                passes=passes,
            )

    try:
        results = await asyncio.gather(
            *(rewrite_one(index, text) for index, text in enumerate(request.texts))
        )
        results.sort(key=lambda item: item.index)
        return BatchRewriteResponse(
            results=results,
            count=len(results),
            workers=MODEL_WORKERS,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量重写失败: {str(e)}")


@app.api_route("/rewrite/simple", methods=["GET", "POST"])
async def rewrite_simple(text: str, temperature: float = 0.7):
    """
    简单重写接口 (GET参数方式)
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="文本不能为空")

    try:
        rewritten, similarity, passes = await run_rewrite(text, temperature, 2048, DEFAULT_STRENGTH)

        return {
            "result": rewritten,
            "strength": DEFAULT_STRENGTH,
            "similarity": similarity,
            "passes": passes,
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重写失败: {str(e)}")


def main():
    """启动服务"""
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=1002,
        log_level="info"
    )


if __name__ == "__main__":
    main()
