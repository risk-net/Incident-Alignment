# pairwise_and_clustering

## 用途

负责 pairwise 二分类训练与图聚类解码，是事件对齐核心建模模块。

## 脚本

- `pairwise_data_io.py`：数据读取与 pairwise 样本构造。
- `prepare_pairwise_data.py`：生成 train/dev/test 缓存数据。
- `pairwise_model.py`：Deep+Wide 模型与推理工具。
- `train_deepwide_pairwise.py`：current 架构训练入口。
- `graph_clustering.py`：图聚类与评估逻辑。
- `sparse_complete_linkage.py`：面向全量数据的 sparse complete-link 聚类（不构建稠密距离矩阵）。
- `decode_graph_from_pairwise_checkpoints.py`：checkpoint 解码与配置搜索评估。

## 输入

- prepared pairwise 数据。
- embeddings 与召回候选。
- 训练 checkpoint（用于解码阶段）。

## 输出

- 训练日志与模型 checkpoint。
- 解码后的聚类结果与评估指标。

## 运行方式

```bash
python src/Incident_Align_Method/pairwise_and_clustering/prepare_pairwise_data.py
python src/Incident_Align_Method/pairwise_and_clustering/train_deepwide_pairwise.py
python src/Incident_Align_Method/pairwise_and_clustering/decode_graph_from_pairwise_checkpoints.py
```
