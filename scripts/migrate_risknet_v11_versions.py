#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 RiskNet-v1.1 里「漏改的 v1.0 版本表述」改为 v1.1。

范围（只改 A 类）：
  - README.md          / README_zh-CN.md     标题、正文、目录树里的 RiskNet v1.0
  - SOURCE_LICENSES.md                        正文与 Applies to 里的 RiskNet v1.0

不改（B 类，历史记录）：
  - VERSION.md    Supersedes v1.0 / What changed since v1.0 / 对比表里的 v1.0
                  —— 这些是版本历史，改了会自相矛盾

用法：
    python /home/nlper/zlh/Incident-Alignment/scripts/migrate_risknet_v11_versions.py          # 预览
    python /home/nlper/zlh/Incident-Alignment/scripts/migrate_risknet_v11_versions.py --apply  # 执行
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path("/data2/risknet_data_release/RiskNet-v1.1")

# 只处理这几个文件；VERSION.md 明确排除（保留版本历史）
TARGETS = ["README.md", "README_zh-CN.md", "SOURCE_LICENSES.md"]

# 精确的替换对：只替换「版本号语境」下的 v1.0，不碰正文里别的 1.0
REPLACEMENTS = [
    # 标题与正文
    ("# RiskNet v1.0", "# RiskNet v1.1"),
    ("RiskNet v1.0 release structure", "RiskNet v1.1 release structure"),
    ("按 RiskNet v1.0 发布目录结构组织", "按 RiskNet v1.1 发布目录结构组织"),
    ("**RiskNet v1.0**", "**RiskNet v1.1**"),
    ("RiskNet v1.0 public release", "RiskNet v1.1 public release"),
    ("preparing RiskNet v1.0", "preparing RiskNet v1.1"),
    ("**Applies to:** RiskNet v1.0", "**Applies to:** RiskNet v1.1"),
    # 目录树里的目录名
    ("RiskNet-v1.0/", "RiskNet-v1.1/"),
]


def patch_text(text: str):
    """返回 (新文本, 命中列表)。"""
    hits = []
    for old, new in REPLACEMENTS:
        if old in text:
            n = text.count(old)
            text = text.replace(old, new)
            hits.append((old, new, n))
    return text, hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="实际写入（默认只预览）")
    ap.add_argument("--no-backup", action="store_true", help="不生成 .bak 备份")
    args = ap.parse_args()

    if not ROOT.is_dir():
        sys.exit(f"❌ 目录不存在：{ROOT}")

    total = 0
    for name in TARGETS:
        p = ROOT / name
        if not p.is_file():
            print(f"跳过（不存在）：{name}")
            continue
        orig = p.read_text(encoding="utf-8")
        new, hits = patch_text(orig)
        if not hits:
            print(f"{name}: 无匹配，跳过")
            continue

        print(f"\n=== {name} ===")
        for old, nw, n in hits:
            print(f"  ×{n}  {old!r}\n     → {nw!r}")
        total += sum(h[2] for h in hits)

        if args.apply:
            if not args.no_backup:
                bak = p.with_suffix(p.suffix + ".bak")
                shutil.copy2(p, bak)
                print(f"  备份 → {bak.name}")
            p.write_text(new, encoding="utf-8")
            print("  已写入")
        else:
            print("  （预览模式，未写入）")

    print(f"\n共待替换 {total} 处。")
    if not args.apply:
        print("确认无误后加 --apply 执行。")
    else:
        print("\n校验：")
        for name in TARGETS:
            p = ROOT / name
            if not p.is_file():
                continue
            txt = p.read_text(encoding="utf-8")
            left = txt.count("RiskNet v1.0") + txt.count("RiskNet-v1.0")
            print(f"  {name}: 剩余 RiskNet v1.0 / v1.0 目录 = {left}"
                  + ("  ✅" if left == 0 else "  ⚠️"))
        v = ROOT / "VERSION.md"
        if v.is_file():
            print(f"  VERSION.md 未改动，仍含 v1.0 历史表述 "
                  f"({v.read_text(encoding='utf-8').count('v1.0')} 处) —— 符合预期")


if __name__ == "__main__":
    main()
