#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全链路全量推理集成脚本。

按顺序执行完整流程:
  1. Step 1: 从数据库生成 embedding + 构建 FAISS 索引
  2. Step 2: 双路召回 (text + event FAISS)
  3. Step 3: pairwise 推理（产出 pair_predictions_full.jsonl）
  4. Step 4: complete-link 聚类（从 pair 推理结果读回分数）

用法:
  export DB_PASSWORD=<密码>
  python src/Incident_Align_Method/full_application/run_full_pipeline.py \
      --config config/Incident_Align_Method-full_application-config.ini
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = BASE_DIR / "config" / "Incident_Align_Method-full_application-config.ini"

# 子脚本（相对项目根目录）
STEP_SCRIPTS = [
    {
        "name": "Step 1: Embedding + FAISS",
        "script": "src/Incident_Align_Method/full_application/build_db_embeddings_and_faiss.py",
    },
    {
        "name": "Step 2: Dual Recall",
        "script": "src/Incident_Align_Method/full_application/run_full_dual_recall.py",
    },
    {
        "name": "Step 3: Pairwise Inference",
        "script": "src/Incident_Align_Method/full_application/run_full_inference.py",
    },
    {
        "name": "Step 4: Clustering",
        "script": "src/Incident_Align_Method/full_application/run_full_clustering.py",
    },
]


def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def run_step(script_rel: str, config_path: str) -> bool:
    """运行单个子脚本，返回是否成功。"""
    script = str(BASE_DIR / script_rel)
    log(f"🚀 执行 {script_rel}")
    log(f"   命令: python {script} --config {config_path}")

    result = subprocess.run(
        [sys.executable, script, "--config", config_path],
        cwd=str(BASE_DIR),
    )

    if result.returncode == 0:
        log(f"✅ {script_rel} 成功完成\n")
        return True
    else:
        log(f"❌ {script_rel} 失败 (exit code {result.returncode})\n")
        return False


def main():
    parser = argparse.ArgumentParser(description="全链路全量推理集成脚本")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG),
                        help="INI 配置文件路径")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.exists(config_path):
        log(f"❌ 配置文件不存在: {config_path}")
        sys.exit(1)

    # 检查 DB_PASSWORD
    if not os.environ.get("DB_PASSWORD"):
        log("❌ 环境变量 DB_PASSWORD 未设置")
        log("   请先执行: export DB_PASSWORD=<你的密码>")
        sys.exit(1)

    log("=" * 70)
    log("全链路全量推理")
    log("=" * 70)
    log(f"配置文件: {config_path}")

    # 执行三个步骤
    total_start = time.time()
    for i, step in enumerate(STEP_SCRIPTS, 1):
        log("\n" + "=" * 70)
        log(f"{step['name']} ({i}/{len(STEP_SCRIPTS)})")
        log("=" * 70)

        step_start = time.time()
        success = run_step(step["script"], config_path)
        elapsed = time.time() - step_start

        log(f"⏱  {step['name']} 耗时: {elapsed / 3600:.1f} 小时")

        if not success:
            log("\n" + "=" * 70)
            log(f"❌ 流程在 {step['name']} 处失败，终止执行")
            log("=" * 70)
            sys.exit(1)

    # 全部成功
    total_elapsed = time.time() - total_start
    log("\n" + "=" * 70)
    log("✅ 全链路推理全部完成")
    log(f"   总耗时: {total_elapsed / 3600:.1f} 小时")
    log("=" * 70)

    log("\n🎉 全量重跑完成。输出结构:")
    log(f"   <artifacts_root>/embeddings/      # embedding")
    log(f"   <artifacts_root>/faiss/           # FAISS 索引")
    log(f"   <artifacts_root>/recall/          # 召回候选")
    log(f"   <artifacts_root>/full_inference/  # 推理结果 + 聚类结果")


if __name__ == "__main__":
    main()
