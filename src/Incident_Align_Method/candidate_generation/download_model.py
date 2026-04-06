#!/usr/bin/env python3
"""
BGE-M3模型下载脚本
下载BAAI/bge-m3模型到本地目录或sentence-transformers缓存
"""

import os
from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[3]
OUTPUT_DIR = os.path.join(BASE_DIR, "models", "bge-m3")
USE_CACHE = True

def download_bge_model(output_dir=None, use_cache=True):
    """
    下载BGE-M3模型
    Args:
        output_dir: 指定下载目录，如果为None则使用sentence-transformers缓存
        use_cache: 是否使用缓存
    """
    print("开始下载BGE-M3模型...")

    try:
        if output_dir:
            output_path = output_dir
            os.makedirs(output_path, exist_ok=True)
            print(f"📁 下载到指定目录: {output_path}")

            config_file = os.path.join(output_path, "config_sentence_transformers.json")
            if use_cache and os.path.exists(config_file):
                print("♻️ 检测到本地已存在模型，直接复用")
                model = SentenceTransformer(output_path)
            else:
                model = SentenceTransformer("BAAI/bge-m3")
                model.save(output_path)
            model_path = output_path
        else:
            # 使用默认缓存目录
            print("📁 使用sentence-transformers默认缓存目录")
            model = SentenceTransformer("BAAI/bge-m3")
            # 获取缓存路径
            import torch
            cache_dir = torch.hub.get_dir()  # 通常是~/.cache/torch/hub
            model_path = os.path.join(cache_dir, "checkpoints", "sentence-transformers_BAAI_bge-m3")

        # 测试编码
        test_embedding = model.encode(["test"])
        print(f"✅ 模型下载并加载成功!")
        print(f"📁 模型路径: {model_path}")
        print(f"📊 嵌入维度: {test_embedding.shape[1]}")

        return model_path

    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("\n🔧 解决方案:")
        print("1. 设置HuggingFace token:")
        print("   export HF_TOKEN=your_token_here")
        print("   (获取token: https://huggingface.co/settings/tokens)")
        print("")
        print("2. 使用代理:")
        print("   export HTTP_PROXY=http://your-proxy:port")
        print("   export HTTPS_PROXY=http://your-proxy:port")
        print("")
        print("3. 使用镜像:")
        print("   export HF_ENDPOINT=https://hf-mirror.com")
        return None

def main():
    download_bge_model(output_dir=OUTPUT_DIR, use_cache=USE_CACHE)

if __name__ == "__main__":
    main()
