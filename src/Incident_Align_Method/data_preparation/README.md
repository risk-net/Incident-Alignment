# data_preparation

## 用途

负责构建事件对齐评测数据集。

## 脚本

- `generate_dataset.py`：生成评测案例与事件结构文件。

## 输入

- 项目内事件新闻与标注数据源（脚本内按默认路径读取）。

## 输出

- `data/eval_cases.jsonl`
- `data/eval_structure.json`
- 可选纯净版本（`--pure`）

## 运行方式

```bash
python src/Incident_Align_Method/data_preparation/generate_dataset.py
```

```bash
python src/Incident_Align_Method/data_preparation/generate_dataset.py --pure
```
