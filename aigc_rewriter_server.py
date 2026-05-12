#!/usr/bin/env python3
"""
OpenAI-compatible API Server
基于 llama-cpp-python 的本地模型推理服务，提供 OpenAI 兼容接口
"""

import os
import gc
import asyncio
import time
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama
import uvicorn

# 模型配置
MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "qwen3-merged-aigc_zhv3-Q4_K_M.gguf"
MODEL_WORKERS = max(1, int(os.environ.get("LLM_WORKERS", "1")))
N_GPU_LAYERS = int(os.environ.get("LLM_GPU_LAYERS", "-1"))
N_CTX = int(os.environ.get("LLM_N_CTX", "4096"))
GPU_IDLE_TIMEOUT = int(os.environ.get("LLM_IDLE_TIMEOUT", "300"))
SERVER_HOST = os.environ.get("LLM_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("LLM_PORT", "1002"))

# 全局模型状态
llm_pool: list[Llama] = []
llm_queue: Optional[asyncio.Queue[Llama]] = None
llm_loaded: bool = False
last_used_time: float = 0
idle_check_task: Optional[asyncio.Task] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "local"
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 2048
    stream: bool = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]


async def _load_models() -> bool:
    """加载模型到GPU"""
    global llm_pool, llm_queue, llm_loaded, last_used_time

    if llm_loaded:
        return True

    print(f"Loading model... workers={MODEL_WORKERS}, n_gpu_layers={N_GPU_LAYERS}, n_ctx={N_CTX}")
    llm_pool = []
    llm_queue = asyncio.Queue(maxsize=MODEL_WORKERS)

    for _ in range(MODEL_WORKERS):
        model = Llama(
            model_path=str(MODEL_PATH),
            n_gpu_layers=N_GPU_LAYERS,
            n_ctx=N_CTX,
            verbose=False
        )
        llm_pool.append(model)
        llm_queue.put_nowait(model)

    llm_loaded = True
    last_used_time = time.time()
    print(f"Model loaded, workers={len(llm_pool)}")
    return True


async def _release_models():
    """释放GPU资源"""
    global llm_pool, llm_queue, llm_loaded

    if not llm_loaded:
        return

    print("Releasing GPU resources...")

    if llm_queue is not None:
        while not llm_queue.empty():
            try:
                llm_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    for model in llm_pool:
        if hasattr(model, 'close'):
            model.close()

    llm_pool.clear()
    llm_queue = None
    llm_loaded = False
    gc.collect()
    print("GPU resources released")


async def _idle_checker():
    """后台任务：检查空闲时间，超时释放GPU"""
    while True:
        await asyncio.sleep(60)
        if llm_loaded and last_used_time > 0:
            idle_time = time.time() - last_used_time
            if idle_time > GPU_IDLE_TIMEOUT:
                print(f"Idle {idle_time:.0f}s > timeout {GPU_IDLE_TIMEOUT}s, releasing GPU")
                await _release_models()


async def run_completion(messages: list[dict], temperature: float, max_tokens: int) -> str:
    """从模型池取一个空闲实例执行推理"""
    global last_used_time

    if not llm_loaded:
        await _load_models()

    if llm_queue is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    last_used_time = time.time()

    model = await llm_queue.get()
    try:
        response = await asyncio.to_thread(
            model.create_chat_completion,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response["choices"][0]["message"]["content"]
    finally:
        llm_queue.put_nowait(model)


def clean_model_output(content: str) -> str:
    """Remove model reasoning markers from user-facing responses."""
    text = content.strip()
    if text.startswith("<think>"):
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        else:
            text = text.removeprefix("<think>")
    return text.strip()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global idle_check_task

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            f"Download: wget -O {MODEL_PATH} "
            "https://huggingface.co/skskk/aigc-rewriter/resolve/main/qwen3-merged-aigc_zhv3-Q4_K_M.gguf"
        )

    print(f"Server started, GPU idle timeout: {GPU_IDLE_TIMEOUT}s")
    print(f"Model will load on first request, workers={MODEL_WORKERS}")

    idle_check_task = asyncio.create_task(_idle_checker())

    yield

    if idle_check_task:
        idle_check_task.cancel()
    await _release_models()


app = FastAPI(
    title="OpenAI-Compatible API",
    description="Local LLM inference server with OpenAI-compatible interface",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """健康检查"""
    return {
        "status": "ok",
        "model": MODEL_PATH.stem,
        "gpu_loaded": llm_loaded,
        "workers": MODEL_WORKERS,
    }


@app.get("/v1/models")
async def list_models():
    """列出可用模型"""
    return {
        "object": "list",
        "data": [
            {
                "id": "local",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ]
    }


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI兼容的聊天补全接口"""
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    try:
        content = await run_completion(
            messages,
            request.temperature,
            request.max_tokens,
        )
        content = clean_model_output(content)

        return ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=content),
                )
            ],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Completion failed: {str(e)}")


def main():
    """启动服务"""
    uvicorn.run(
        app,
        host=SERVER_HOST,
        port=SERVER_PORT,
        log_level="info"
    )


if __name__ == "__main__":
    main()
