#!/usr/bin/env python3
"""下载 BGE-M3 到本地目录，供 build_embeddings.py 使用。

为什么单独给脚本、而不把权重放进仓库
------------------------------------
BGE-M3 约 2.3 GB，属于第三方模型权重，不随本包分发。研究者需要自行下载一次，
放到 `models/bge-m3/`（`config/Incident_Align_Method-build_embeddings-config.ini`
里的 `model_path` 默认就是它）。

本脚本**不会**自动联网执行——必须由你显式运行。`build_embeddings.py` 使用
`SentenceTransformer(..., local_files_only=True)`，因此模型必须是**本地完整目录**，
而不是 HuggingFace 的 blobs/snapshots 缓存结构。

用法
----
    # 默认走 HuggingFace（需要能访问 huggingface.co）
    python scripts/download_bge_m3.py

    # 大陆网络建议走 ModelScope 镜像（需先 pip install modelscope）
    python scripts/download_bge_m3.py --source modelscope

    # 自定义落盘位置 / 只校验已有模型
    python scripts/download_bge_m3.py --out models/bge-m3
    python scripts/download_bge_m3.py --verify-only

下载完成后脚本会：
  1. 检查 SentenceTransformer / tokenizer / 权重文件是否齐全；
  2. 打印总体积与关键文件 SHA-256；
  3. 写出 `models/bge-m3/DOWNLOAD_MANIFEST.json`，便于随包登记来源与校验。
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ID = "BAAI/bge-m3"

# SentenceTransformer 本地目录至少需要这些
REQUIRED_FILES = [
    "config.json",
    "modules.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "sentence_bert_config.json",
    "config_sentence_transformers.json",
]
WEIGHT_CANDIDATES = ["model.safetensors", "pytorch_model.bin"]


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dir_size(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def verify(out: Path):
    problems = [f for f in REQUIRED_FILES if not (out / f).is_file()]
    weights = [w for w in WEIGHT_CANDIDATES if (out / w).is_file()]
    if not weights:
        problems.append("缺少权重文件（model.safetensors 或 pytorch_model.bin）")
    if problems:
        print("❌ 模型目录不完整：")
        for p in problems:
            print(f"   - {p}")
        return False

    size = dir_size(out)
    print(f"✅ 模型目录完整：{out}")
    print(f"   体积：{size / 2**30:.2f} GiB")
    print(f"   权重：{weights[0]}  sha256={sha256_file(out / weights[0])[:16]}…")
    for f in ("config.json", "modules.json"):
        print(f"   {f}  sha256={sha256_file(out / f)[:16]}…")
    return True


def download_hf(out: Path):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("缺少 huggingface_hub：pip install huggingface_hub")

    print(f"从 HuggingFace 下载 {REPO_ID} -> {out}")
    print("（首次约 2.3 GiB；支持断点续传，重复运行只会补齐缺失文件）")
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(out),
        # 只取 SentenceTransformer 运行所需，避免多余格式
        allow_patterns=[
            "*.json", "*.txt", "*.model", "*.safetensors", "*.bin",
            "1_Pooling/*", "sentence_bert_config.json", "modules.json",
        ],
    )


def download_modelscope(out: Path):
    try:
        from modelscope import snapshot_download as ms_download
    except ImportError:
        sys.exit("缺少 modelscope：pip install modelscope")

    print(f"从 ModelScope 下载 {REPO_ID} …")
    cache_dir = ms_download(REPO_ID)
    print(f"ModelScope 缓存目录：{cache_dir}")
    print(f"正在复制到 {out} …")
    import shutil
    out.mkdir(parents=True, exist_ok=True)
    for src in Path(cache_dir).rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(cache_dir)
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser(description="下载 BGE-M3 到 models/bge-m3")
    ap.add_argument("--out", default="models/bge-m3", help="本地目标目录")
    ap.add_argument("--source", choices=["huggingface", "modelscope"], default="huggingface")
    ap.add_argument("--verify-only", action="store_true", help="只校验已有目录，不下载")
    a = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    out = Path(a.out)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)

    if a.verify_only:
        sys.exit(0 if verify(out) else 1)

    if a.source == "huggingface":
        download_hf(out)
    else:
        download_modelscope(out)

    if not verify(out):
        sys.exit("下载后校验失败，请重试或改用另一种 source")

    manifest = {
        "repo_id": REPO_ID,
        "source": a.source,
        "local_dir": str(out),
        "bytes": dir_size(out),
        "files": {
            str(p.relative_to(out)): {"sha256": sha256_file(p), "bytes": p.stat().st_size}
            for p in sorted(out.rglob("*")) if p.is_file()
        },
        "note": "第三方模型权重，不随仓库分发；由 scripts/download_bge_m3.py 获取。",
    }
    (out / "DOWNLOAD_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n清单 -> {out / 'DOWNLOAD_MANIFEST.json'}")
    print("\n下一步：")
    print("  python research_pipeline/Incident_Align_Method/candidate_generation/build_embeddings.py \\")
    print("      --config config/Incident_Align_Method-build_embeddings-config.ini")


if __name__ == "__main__":
    main()
