#!/usr/bin/env bash
# =============================================================================
# 全量自检脚本（研究者在本机执行，不由本包自动运行）
#
# 用途：在不重跑训练/推理的前提下，验证发布包自身是否自洽。
#
# 用法：
#   bash scripts/run_full_validation.sh                 # 标准自检
#   bash scripts/run_full_validation.sh --with-theta     # 额外重算 Dev 阈值（较慢）
#   PYTHON=/path/to/python bash scripts/run_full_validation.sh
#
# 退出码：0 = 全部通过；非 0 = 有检查项失败（详见输出）
# =============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
WITH_THETA=0
[ "${1:-}" = "--with-theta" ] && WITH_THETA=1

PASS=0; FAIL=0; SKIP=0
ok()   { echo "  ✅ $*"; PASS=$((PASS+1)); }
bad()  { echo "  ❌ $*"; FAIL=$((FAIL+1)); }
skip() { echo "  ⏭️  $*"; SKIP=$((SKIP+1)); }
hdr()  { echo; echo "── $* ─────────────────────────────────────────────"; }

echo "仓库根目录: $ROOT"
echo "Python     : $($PYTHON -V 2>&1 || echo '不可用')"

# ---------------------------------------------------------------- 1. 环境
hdr "1. 运行环境"
$PYTHON - <<'EOF' && ok "python / numpy / scipy / sklearn / torch 可导入" || bad "依赖导入失败"
import sys, numpy, scipy, sklearn
print(f"    python={sys.version.split()[0]} numpy={numpy.__version__} scipy={scipy.__version__} sklearn={sklearn.__version__}")
try:
    import torch; print(f"    torch={torch.__version__} cuda={torch.cuda.is_available()}")
except Exception as e:
    raise SystemExit(f"    torch 导入失败: {e}")
EOF

$PYTHON -c 'import pytest' 2>/dev/null && ok "pytest 可用" || skip "pytest 未安装（pip install pytest 后可跑单元测试）"

# ---------------------------------------------------------------- 2. 单元测试
hdr "2. 单元测试 (tests/)"
if $PYTHON -c 'import pytest' 2>/dev/null; then
    if PYTHONPATH=src $PYTHON -m pytest -q tests 2>&1 | tail -3; then
        ok "pytest 通过"
    else
        bad "pytest 有失败（已知：test_core::test_invalid_partition 2 例）"
    fi
else
    skip "跳过（pytest 未安装）"
fi

# ---------------------------------------------------------------- 3. 资产放置
hdr "3. 资产放置 (check_assets.py)"
for profile in chinese english train folds full; do
    if $PYTHON scripts/check_assets.py --root . --profile "$profile" >/tmp/ca_$profile.json 2>&1; then
        ok "profile=$profile"
    else
        bad "profile=$profile 缺文件: $(cat /tmp/ca_$profile.json | tr -d '\n' | cut -c1-160)"
    fi
done

# ---------------------------------------------------------------- 4. Dev 分数与阈值
hdr "4. Dev 分数 JSONL 与锁阈值 schema"
for i in 01 02 03 04 05; do
    f="outputs/dev/fold_${i}_scores.jsonl"
    [ -s "$f" ] && ok "$f ($(wc -l < "$f") 行)" || bad "$f 缺失或为空"
done
$PYTHON - <<'EOF' && ok "locked_thetas.json schema 正确且五折齐全" || bad "locked_thetas.json schema 不符"
import json, sys
d = json.load(open('outputs/dev/locked_thetas.json'))
assert d.get('source') == 'dev_only', f"source={d.get('source')!r} 应为 'dev_only'"
lt = d.get('locked_thetas') or {}
assert set(lt) == {'1','2','3','4','5'}, f"locked_thetas 折号不齐: {sorted(lt)}"
print(f"    thetas={lt}")
EOF
for i in 01 02 03 04 05; do
    f="outputs/dev/fold_${i}_scan.json"
    $PYTHON -c "
import json,sys
d=json.load(open('$f'))
sys.exit(0 if d.get('source')=='dev_only' and 'selected_theta' in d else 1)" 2>/dev/null \
        && ok "$f (theta=$( $PYTHON -c "import json;print(json.load(open('$f'))['selected_theta'])" ))" \
        || bad "$f 缺少 source=dev_only 或 selected_theta"
done

# ---------------------------------------------------------------- 5. Gold 自洽
hdr "5. Gold 与 fold splits 自洽"
$PYTHON - <<'EOF' && ok "中文 Gold 1,283 篇 / 185 事件" || bad "中文 Gold 不符"
import csv
rows = list(csv.DictReader(open('data/chinese_gold/gold_v2.0.2_mapping.csv', encoding='utf-8-sig')))
ids = [r['report_id'].strip() for r in rows]
evs = [r['gold_v2_event_id'].strip() for r in rows]
assert len(rows)==1283 and len(set(ids))==1283 and len(set(evs))==185
assert set(evs)=={f'zh_gold_{i:04d}' for i in range(1,186)}
print('    1,283 reports / 185 events / 连续事件ID')
EOF
$PYTHON - <<'EOF' && ok "英文 5 折 test_gold 拼接 = 5,998 篇 / 1,742 事件且互不重叠" || bad "英文每折 Gold 不自洽"
import csv
seen_r, seen_e = set(), set()
for i in range(1,6):
    rows = list(csv.DictReader(open(f'data/english/folds/fold_{i:02d}/test_gold.csv', encoding='utf-8')))
    r = {x['report_id'] for x in rows}; e = {x['gold_v2_event_id'] for x in rows}
    assert not (seen_r & r) and not (seen_e & e), f'fold_{i:02d} 与前面折重叠'
    seen_r |= r; seen_e |= e
assert len(seen_r)==5998 and len(seen_e)==1742, (len(seen_r), len(seen_e))
print('    5,998 reports / 1,742 events，跨折零重叠')
EOF

# ---------------------------------------------------------------- 6. 校验和
hdr "6. 校验和"
[ -f SHA256SUMS.txt ] || bad "SHA256SUMS.txt 缺失"
if [ -f SHA256SUMS.txt ]; then
    n_fail=$(sha256sum -c SHA256SUMS.txt 2>/dev/null | grep -vc '成功\|OK$')
    if [ "$n_fail" -eq 0 ]; then ok "SHA256SUMS.txt 全部匹配"; else bad "SHA256SUMS.txt 有 $n_fail 项不匹配（数据变更后需重生成）"; fi
fi
[ -f docs/ASSET_MANIFEST.json ] && ok "docs/ASSET_MANIFEST.json 存在" || bad "docs/ASSET_MANIFEST.json 缺失"

# ---------------------------------------------------------------- 7. 可选：重算阈值
hdr "7. （可选）Dev 阈值重算复现"
if [ "$WITH_THETA" -eq 1 ]; then
    for i in 01 02 03 04 05; do
        PYTHONPATH=src $PYTHON -m incident_alignment.workflows.cli select-dev \
            --gold "data/english/folds/fold_${i}/dev/gold.csv" \
            --scores "outputs/dev/fold_${i}_scores.jsonl" \
            --out "/tmp/rescan_${i}.json" --min 0.10 --max 0.95 --step 0.01 >/dev/null 2>&1
        got=$($PYTHON -c "import json;print(json.load(open('/tmp/rescan_${i}.json'))['selected_theta'])" 2>/dev/null)
        want=$($PYTHON -c "import json;print(json.load(open('outputs/dev/locked_thetas.json'))['locked_thetas']['$((10#$i))'])")
        [ "$got" = "$want" ] && ok "fold_$i theta=$got 与锁定值一致" || bad "fold_$i 重算=$got 锁定=$want"
    done
else
    skip "跳过（加 --with-theta 启用，约需数分钟）"
fi

# ---------------------------------------------------------------- 汇总
echo
echo "════════════════════════════════════════════════════════════"
printf "  通过 %d 项 | 失败 %d 项 | 跳过 %d 项\n" "$PASS" "$FAIL" "$SKIP"
[ "$FAIL" -eq 0 ] && echo "  结论：全部通过 ✅" || echo "  结论：存在失败项，见上方 ❌ ❌"
echo "════════════════════════════════════════════════════════════"
exit $([ "$FAIL" -eq 0 ] && echo 0 || echo 1)
