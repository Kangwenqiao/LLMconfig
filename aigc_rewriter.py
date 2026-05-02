#!/usr/bin/env python3
"""
AIGC Rewriter - 降AIGC重写模型
基于 skskk/aigc-rewriter GGUF 模型
"""

import os
from pathlib import Path
from llama_cpp import Llama

# 模型配置
MODEL_URL = "https://huggingface.co/skskk/aigc-rewriter/resolve/main/qwen3-merged-aigc_zhv3-Q4_K_M.gguf"
MODEL_DIR = Path.home() / ".cache" / "aigc-rewriter"
MODEL_PATH = MODEL_DIR / "qwen3-merged-aigc_zhv3-Q4_K_M.gguf"


def download_model():
    """下载模型文件"""
    if MODEL_PATH.exists():
        print(f"模型已存在: {MODEL_PATH}")
        return True

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("正在下载模型...")
    import urllib.request

    # 使用代理环境变量
    proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    if proxy:
        print(f"使用代理: {proxy}")

    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"模型下载完成: {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"下载失败: {e}")
        print("请手动下载模型或使用代理:")
        print(f"  proxychains4 wget -O {MODEL_PATH} {MODEL_URL}")
        return False


class AIGCRewriter:
    """AIGC重写器"""

    def __init__(self, n_gpu_layers: int = -1, n_ctx: int = 4096):
        """
        初始化重写器

        Args:
            n_gpu_layers: GPU层数 (-1表示全部加载到GPU, 0表示仅CPU)
            n_ctx: 上下文长度
        """
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"模型文件不存在: {MODEL_PATH}\n"
                "请先运行: python aigc_rewriter.py --download"
            )

        print("正在加载模型...")
        self.llm = Llama(
            model_path=str(MODEL_PATH),
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=False
        )
        print("模型加载完成")

    def rewrite(self, text: str, temperature: float = 0.7, max_tokens: int = 2048) -> str:
        """
        重写文本

        Args:
            text: 需要重写的文本
            temperature: 生成温度
            max_tokens: 最大生成token数

        Returns:
            重写后的文本
        """
        prompt = f"""请对以下文本进行重写，保持原意的同时使表达更加自然流畅：

{text}

重写后的文本："""

        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|im_end|>", "\n\n\n"],
            echo=False
        )

        return response["choices"][0]["text"].strip()

    def chat_rewrite(self, text: str, temperature: float = 0.7, max_tokens: int = 2048) -> str:
        """
        使用对话模式重写文本

        Args:
            text: 需要重写的文本
            temperature: 生成温度
            max_tokens: 最大生成token数

        Returns:
            重写后的文本
        """
        messages = [
            {"role": "system", "content": "你是一个专业的文本重写助手，擅长将AI生成的内容改写为更加自然、人类化的表达。"},
            {"role": "user", "content": f"请重写以下文本，保持原意但使表达更自然：\n\n{text}"}
        ]

        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|im_end|>"]
        )

        return response["choices"][0]["message"]["content"].strip()

    def __del__(self):
        """释放资源"""
        if hasattr(self, 'llm'):
            del self.llm


def interactive_mode():
    """交互模式"""
    print("=" * 50)
    print("AIGC Rewriter - 降AIGC重写工具")
    print("=" * 50)

    rewriter = AIGCRewriter()

    print("\n输入文本进行重写，输入 'quit' 或 'exit' 退出")
    print("-" * 50)

    while True:
        try:
            text = input("\n请输入文本: ").strip()

            if text.lower() in ['quit', 'exit', 'q']:
                print("再见!")
                break

            if not text:
                print("请输入有效文本")
                continue

            print("\n重写中...")
            result = rewriter.chat_rewrite(text)
            print(f"\n重写结果:\n{result}")

        except KeyboardInterrupt:
            print("\n\n已中断，再见!")
            break
        except Exception as e:
            print(f"错误: {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="AIGC Rewriter - 降AIGC重写工具")
    parser.add_argument("--download", action="store_true", help="下载模型")
    parser.add_argument("--text", type=str, help="要重写的文本")
    parser.add_argument("--file", type=str, help="从文件读取文本")
    parser.add_argument("--output", type=str, help="输出到文件")
    parser.add_argument("--cpu", action="store_true", help="仅使用CPU")
    parser.add_argument("--temperature", type=float, default=0.7, help="生成温度")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互模式")

    args = parser.parse_args()

    if args.download:
        download_model()
        return

    if not MODEL_PATH.exists():
        print("模型不存在，正在下载...")
        if not download_model():
            return

    # GPU层数设置
    n_gpu_layers = 0 if args.cpu else -1

    rewriter = AIGCRewriter(n_gpu_layers=n_gpu_layers)

    try:
        if args.interactive:
            interactive_mode()
        elif args.text:
            result = rewriter.chat_rewrite(args.text, temperature=args.temperature)
            print(result)
            if args.output:
                Path(args.output).write_text(result, encoding="utf-8")
                print(f"\n已保存到: {args.output}")
        elif args.file:
            text = Path(args.file).read_text(encoding="utf-8")
            result = rewriter.chat_rewrite(text, temperature=args.temperature)
            print(result)
            if args.output:
                Path(args.output).write_text(result, encoding="utf-8")
                print(f"\n已保存到: {args.output}")
        else:
            interactive_mode()
    finally:
        # 显式释放资源
        del rewriter
        print("\nGPU资源已释放")


if __name__ == "__main__":
    main()
