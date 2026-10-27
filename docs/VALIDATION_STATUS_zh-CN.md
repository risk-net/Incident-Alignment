# 验证状态

> 📖 English version: [VALIDATION_STATUS.md](./VALIDATION_STATUS.md)

最后更新：2026-10-27。

## 已实际执行的检查

| 检查项 | 命令 | 结果 |
|---|---|---|
| 基线自检（不训练、不推理） | `bash scripts/run_full_validation.sh` | **20 通过 / 1 失败 / 3 跳过** |
| 资产放置（全部 profile） | `python scripts/check_assets.py --profile {chinese,english,train,folds,full}` | **5/5 通过** |
| Dev 概率 → M6 JSONL | `python scripts/npz_to_score_jsonl.py --all ...` | 5/5 折转换完成（60,731 / 50,735 / 55,635 / 53,035 / 55,354 对） |
| Dev 选阈 + 锁定 | `incident-align select-dev` ×5 → `lock-thetas` | **精确复现**锁定阈值：0.52 / 0.45 / 0.59 / 0.63 / 0.61 |
| 英文每折 Gold 自洽 | 5 份 `test_gold.csv` 并集 | 5,998 篇 / 1,742 事件，跨折零重叠 |
| 中文 Gold 结构 | `gold_v2.0.2_mapping.csv` | 1,283 篇 / 185 事件，`zh_gold_0001…0185` 连续 |
| 单元测试 | `PYTHONPATH=src python -m pytest -q tests` | **40 通过 / 0 失败** |

基线自检里那 1 项失败是**重生成之前**的 `SHA256SUMS.txt`；执行
`python scripts/regen_checksums.py` 后即通过。

## 已知失败 / 局限

1. ~~`tests/test_core.py::test_invalid_partition` 有 2 例失败。~~ **已修复 / 不再复现。**
   `canonicalize_partition` 现在会拒绝空簇与重复成员（跨簇与簇内重复均拒绝），
   三个参数化用例都按测试期望抛出 `ValueError`。当前测试结果：**40 通过 / 0 失败**。

2. **本包验证未执行 GPU 推理。** 上述选阈与评测均在 CPU 上用冻结的 Dev `.npz`
   缓存完成，没有重新跑 DeepWide 前向。要闭合这一环，需研究者侧对 5 个 checkpoint
   重跑 `export_pair_scores.py`。

3. **英文 held-out 评测必须使用每折 Gold。** 英文五折的 Test 集互不重叠
   （各约 1,200 篇），必须用 `incident-align evaluate-folds --fold-gold`。若用单份全量
   5,998 篇 Gold：实测 fold_01 预测簇数从 226 虚增到 5,024，合成分数下 Hungarian G→P
   被稀释（≈0.57 → ≈0.63）。中文因五折打分的是同一份 1,283 篇域，所以用单份 `--gold`。

4. **`scripts/export_pair_scores.py` 默认 `--topm 200`**，与英文训练协议
   （`TOPM_OUT = 200`）一致；中文评测使用 `--topm 250`。传其他值会静默改变被评分的
   pair 集合，使锁定阈值失效。`docs/REPRODUCIBILITY_zh-CN.md` 中的英文命令现已显式
   传入 `--topm 200`。

5. **随包 5 个 checkpoint 携带旧的 `prepared_dir` 元数据**（历史仓库路径）。
   `decode_graph_from_pairwise_checkpoints.py` 会校验该字段与配置一致，因此在当前布局下
   运行它需要先用现在的训练脚本重训。

## 复现本状态

```bash
python -m pip install -e '.[test,deepwide]'
python -m pip install pytest
bash scripts/run_full_validation.sh --with-theta
```
