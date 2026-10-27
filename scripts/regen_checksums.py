#!/usr/bin/env python3
"""重生成发布校验物：SHA256SUMS.txt 与 docs/ASSET_MANIFEST.json。

何时运行
--------
任何会改变随包文件的操作之后，例如：
  - 跑完 P1-6（补 pred_clusters_test.json）
  - 重建 embedding / recall / checkpoint
  - 修改文档或脚本

用法
----
    python scripts/regen_checksums.py            # 重生成两者
    python scripts/regen_checksums.py --check    # 只校验，不写入

约定
----
- 覆盖范围：仓库内所有常规文件，**排除** .git/、__pycache__/、.pytest_cache/
  以及 SHA256SUMS.txt / ASSET_MANIFEST.json 自身。
- SHA256SUMS.txt：标准 `sha256sum -c` 格式，路径相对仓库根。
- ASSET_MANIFEST.json：按类别（data / checkpoints / outputs / docs / code …）
  汇总，含字节数、文件数与逐文件哈希。
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

EXCLUDE_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv", "node_modules"}
EXCLUDE_NAMES = {"SHA256SUMS.txt", "ASSET_MANIFEST.json"}
SELF_REL = {"scripts/regen_checksums.py"}

# 本地实验产物不算发布资产：它们是运行基线实验的副产物，体积大且因人而异，
# 不应进入 ASSET_MANIFEST / SHA256SUMS（也不会进 git，见 .gitignore 的 outputs/*）。
# 注意：outputs/chinese/、outputs/english/、outputs/dev/ 里是**正式流水线产物**，
# 是论文数值的来源，必须继续纳入清单——只排除基线实验目录。
EXCLUDE_PATH_PREFIXES = (
    "outputs/benchmarks/",   # LR 基线实验（特征缓存、分数、日志、对照表）
    ".smoke/",               # 冒烟测试残留
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def category_of(rel: str) -> str:
    if rel.startswith("data/"):
        return "data"
    if rel.startswith("checkpoints/"):
        return "checkpoints"
    if rel.startswith("outputs/"):
        return "outputs"
    if rel.startswith("docs/"):
        return "docs"
    if rel.startswith("config/"):
        return "config"
    if rel.startswith("scripts/"):
        return "scripts"
    if rel.startswith("src/") or rel.startswith("research_pipeline/"):
        return "code"
    if rel.startswith("tests/"):
        return "tests"
    return "root"


def collect(root: Path):
    rows = []
    for p in sorted(root.rglob("*")):
        if p.is_dir():
            continue
        rel = p.relative_to(root).as_posix()
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        if p.name in EXCLUDE_NAMES:
            continue
        if any(rel.startswith(pfx) for pfx in EXCLUDE_PATH_PREFIXES):
            continue
        if p.stat().st_size == 0:
            continue
        rows.append({
            "path": rel,
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p),
            "category": category_of(rel),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--check", action="store_true", help="只校验不写入")
    a = ap.parse_args()

    root = Path(a.root).resolve()
    rows = collect(root)
    total = sum(r["bytes"] for r in rows)

    by_cat = {}
    for r in rows:
        c = by_cat.setdefault(r["category"], {"files": 0, "bytes": 0})
        c["files"] += 1
        c["bytes"] += r["bytes"]

    print(f"扫描到 {len(rows)} 个文件，合计 {total/2**30:.2f} GiB")
    for c in sorted(by_cat):
        v = by_cat[c]
        print(f"  {c:12s} {v['files']:5d} 个  {v['bytes']/2**30:8.2f} GiB")

    if a.check:
        sums = root / "SHA256SUMS.txt"
        if not sums.exists():
            sys.exit("SHA256SUMS.txt 不存在")
        current = {}
        for line in sums.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            h, _, rel = line.partition("  ")
            current[rel.strip()] = h.strip()
        now = {r["path"]: r["sha256"] for r in rows}
        added = sorted(set(now) - set(current))
        removed = sorted(set(current) - set(now))
        changed = sorted(k for k in set(now) & set(current) if now[k] != current[k])
        print(f"\n新增 {len(added)} | 删除 {len(removed)} | 变更 {len(changed)}")
        for label, items in (("新增", added), ("删除", removed), ("变更", changed)):
            for it in items[:10]:
                print(f"  {label}: {it}")
        sys.exit(0 if not (added or removed or changed) else 1)

    # SHA256SUMS.txt
    lines = [f'{r["sha256"]}  {r["path"]}' for r in rows]
    (root / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n已写入 SHA256SUMS.txt（{len(lines)} 条）")

    # ASSET_MANIFEST.json
    manifest = {
        "note": "由 scripts/regen_checksums.py 生成；记录随包资产的分类、字节数与 SHA-256。"
                "大文件（embedding/checkpoint/recall）如需单独托管，以本清单为准。",
        "totals": {"files": len(rows), "bytes": total, "gib": round(total / 2**30, 3)},
        "by_category": by_cat,
        "files": rows,
    }
    out = root / "docs" / "ASSET_MANIFEST.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"已写入 {out.relative_to(root)}")


if __name__ == "__main__":
    main()
