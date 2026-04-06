
#!/usr/bin/env python3
"""
事件对齐评估数据集生成脚本 (Event Alignment Evaluation Dataset Generator)

此脚本用于生成事件对齐方法直接可消费的评估数据集，包含：
1. eval_cases.jsonl - 事件级新闻案例（含标注信息）
2. eval_structure.json - 事件结构信息

可选生成纯净版本：
- eval_cases_pure.jsonl - 只包含有事件对应关系的案例

使用方法：
# 生成标准版本（包含所有事件级新闻）
python generate_dataset.py

# 生成纯净版本（只包含有事件对应关系的新闻）
python generate_dataset.py --pure

依赖：
- Python 3.6+
- pathlib
- json
- argparse

"""

import argparse
import glob
import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]


def write_jsonl(output_path, records):
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def extract_standard_event_level_cases_with_annotations(pure_dataset=False):
    """
    提取标准事件级新闻数据集的主函数

    处理流程：
    1. 读取事件索引 (standard_incidents.jsonl)
    2. 读取AIID-AIAAIC标注信息
    3. 合并原始案例和标注信息
    4. 生成输出文件
    """
    print('开始创建标准事件级新闻数据集...')
    # 检查输入文件是否存在
    incidents_file = os.path.join(BASE_DIR, "data", "standard_incidents.jsonl")
    cases_file = os.path.join(BASE_DIR, "data", "standard_cases.jsonl")
    annotations_base = os.path.join(BASE_DIR, "data", "AIID-AIAAIC")

    for input_path in [incidents_file, cases_file]:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"输入文件不存在: {input_path}")
    if not os.path.exists(annotations_base):
        raise FileNotFoundError(f"标注目录不存在: {annotations_base}")


    # 首先读取AIID-AIAAIC中的标注信息，确定哪些案例是真正的事件级
    annotations = {}
    annotated_case_ids = set()
    sources = ['aiid', 'aiaaic']

    print("步骤1: 读取AI标注信息，确定真正的事件级案例...")
    for source in sources:
        event_path = os.path.join(annotations_base, source, 'AIrisk_relevant_event')
        if os.path.exists(event_path):
            print(f'读取 {source} 的标注信息...')
            for json_file in sorted(glob.glob(os.path.join(event_path, '*_result.json'))):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        case_id = int(data.get('case_id', data.get('id', 0)))
                        if case_id > 0:
                            annotations[case_id] = data
                            annotated_case_ids.add(case_id)
                except Exception as e:
                    print(f"处理文件 {json_file} 时出错: {e}")

    print(f'共读取到 {len(annotations)} 个标注案例，这些才是真正的事件级案例')

    # 读取事件索引，只保留包含标注案例的事件
    incidents = []
    event_case_ids_in_incidents = set()

    # 统计去重信息
    total_duplicates_removed = 0
    events_with_duplicates = 0

    print(f"\n步骤2: 读取事件索引，只保留包含标注案例的事件...")

    def to_int(x):
        try:
            return int(x)
        except:
            return None

    with open(incidents_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                incident = json.loads(line.strip())

                # 1) ids 统一转 int + 去掉脏值
                ids_raw = incident.get('ids', [])
                ids_int = [to_int(cid) for cid in ids_raw]
                ids_int = [cid for cid in ids_int if cid is not None]

                # 2) 只保留"事件级新闻"的id（核心）
                kept_ids = [cid for cid in ids_int if cid in annotated_case_ids]

                # 3) 过滤后为空：这个事件不保留
                if not kept_ids:
                    continue

                # 4) 去重处理：强制 ids = sorted(set(ids))
                original_count = len(kept_ids)
                kept_ids_deduped = sorted(set(kept_ids))
                duplicates_removed = original_count - len(kept_ids_deduped)

                if duplicates_removed > 0:
                    events_with_duplicates += 1
                    total_duplicates_removed += duplicates_removed
                    incident_id = incident.get('incident_id', 'unknown')
                    print(f"  事件 {incident_id}: 去重了 {duplicates_removed} 个重复ID")

                # 5) 写回去重后的 ids
                incident['ids'] = kept_ids_deduped

                incidents.append(incident)
                event_case_ids_in_incidents.update(kept_ids_deduped)

            except json.JSONDecodeError as e:
                print(f"警告：第{line_num}行JSON解析错误: {e}")

    print(f'保留了 {len(incidents)} 个事件（每个事件的ids已净化并去重）')
    print(f'这些事件总共包含 {len(event_case_ids_in_incidents)} 个唯一事件级案例')

    # QA报告：去重统计
    if total_duplicates_removed > 0:
        print(f'\\n📊 数据质量报告:')
        print(f'  发现 {events_with_duplicates} 个事件存在内部重复ID')
        print(f'  总共去重了 {total_duplicates_removed} 个重复ID')
        print(f'  平均每个受影响事件去重 {total_duplicates_removed/events_with_duplicates:.1f} 个ID')
    else:
        print('\\n✅ 数据质量报告: 所有事件内部ID都是唯一的')

    # 读取原始案例，只保留有标注信息的案例
    event_level_cases = []
    total_cases = 0

    print(f"\n步骤3: 读取原始案例，只保留有标注信息的案例...")
    with open(cases_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                case = json.loads(line.strip())
                total_cases += 1

                # 只保留有标注信息的案例
                if case['id'] in annotations:
                    # 合并标注信息，保留原始案例的所有字段，并添加标注信息
                    merged_case = case.copy()
                    annotation = annotations[case['id']]

                    # 添加标注相关的字段
                    merged_case['classification_result'] = annotation.get('classification_result')
                    merged_case['ai_tech'] = annotation.get('ai_tech')
                    merged_case['ai_risk'] = annotation.get('ai_risk')
                    merged_case['event_annotation'] = annotation.get('event')  # 重命名为event_annotation以区分

                    # 保留标注中的其他有用字段（如果原始案例中没有）
                    for key, value in annotation.items():
                        if key not in ['case_id', 'id', 'title', 'content', 'text', 'release_date',
                                     'case_link', 'tags', 'location', 'involved_subject', 'new_summary',
                                     'classification_result', 'ai_tech', 'ai_risk', 'event']:
                            if key not in merged_case or merged_case[key] is None:
                                merged_case[key] = value

                    event_level_cases.append(merged_case)

                if total_cases % 1000 == 0:
                    print(f"已处理 {total_cases} 个案例...")

            except json.JSONDecodeError as e:
                print(f"警告：第{line_num}行案例JSON解析错误: {e}")

    print(f'原始案例总数: {total_cases}')
    print(f'事件级案例数: {len(event_level_cases)}')
    print(f'所有事件级案例都有标注信息: {len([c for c in event_level_cases if "classification_result" in c])}')

    # 保存合并后的案例为JSON文件
    if pure_dataset:
        # 纯净版本：只保留在事件结构中有对应关系的案例
        filtered_cases = [case for case in event_level_cases if case['id'] in event_case_ids_in_incidents]
        cases_output_path = os.path.join(BASE_DIR, "data", "eval_cases_pure.jsonl")

        print(f"保存纯净案例文件: {cases_output_path} ({len(filtered_cases)}/{len(event_level_cases)} 案例)")
        write_jsonl(cases_output_path, filtered_cases)
    else:
        # 标准版本：包含所有事件级案例
        cases_output_path = os.path.join(BASE_DIR, "data", "eval_cases.jsonl")
        print(f"保存标准案例文件: {cases_output_path}")
        write_jsonl(cases_output_path, event_level_cases)

    # 保存事件结构信息
    structure_output_path = os.path.join(BASE_DIR, "data", "eval_structure.json")
    print(f"保存结构文件: {structure_output_path}")
    with open(structure_output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'events': incidents,
            'metadata': {
                'total_events': len(incidents),
                'total_cases': len(event_level_cases),
                'total_unique_case_ids': len(event_case_ids_in_incidents),
                'annotated_cases': len(event_level_cases),  # 现在所有案例都有标注信息
                'description': '只包含在AIrisk_relevant_event中有标注信息的案例和对应的事件结构'
            }
        }, f, ensure_ascii=False, indent=2)

    print(f'\n文件创建完成:')
    cases_file_name = os.path.basename(cases_output_path)
    total_cases_count = len(filtered_cases) if pure_dataset else len(event_level_cases)
    print(f'✓ {cases_file_name} - {total_cases_count} 个事件级案例')
    print(f'✓ {os.path.basename(structure_output_path)} - {len(incidents)} 个事件结构')
    print(f'✓ generate_dataset.py - 生成脚本')

    # 验证生成的文件
    print("\n验证生成的文件...")
    try:
        with open(cases_output_path, 'r', encoding='utf-8') as f:
            cases_count = sum(1 for line in f if line.strip())
            print(f"✓ 案例文件包含: {cases_count} 个案例")

        with open(structure_output_path, 'r', encoding='utf-8') as f:
            structure = json.load(f)
            print(f"✓ 结构文件事件数: {len(structure['events'])}")
            print(f"✓ 元数据: {structure['metadata']}")

        # 硬验证：确保每个事件的ids都是事件级新闻
        bad = []
        for inc in structure['events']:
            for cid in inc['ids']:
                if cid not in annotated_case_ids:
                    bad.append((inc.get('incident_id'), cid))

        print(f"硬验证结果 - bad edges: {len(bad)}")
        if bad:
            print(f"❌ 发现非事件级新闻ID: {bad[:10]}...")  # 只显示前10个
            raise AssertionError(f"事件结构中包含 {len(bad)} 个非事件级新闻ID")
        else:
            print("✓ 硬验证通过：所有事件-新闻边都指向事件级新闻")

        # 验证事件内部没有重复ID
        internal_duplicates = []
        for inc in structure['events']:
            ids = inc['ids']
            if len(ids) != len(set(ids)):
                # 找出重复的ID
                from collections import Counter
                id_counts = Counter(ids)
                duplicates = {id: count for id, count in id_counts.items() if count > 1}
                internal_duplicates.append((inc.get('incident_id'), duplicates))

        if internal_duplicates:
            print(f"❌ 发现 {len(internal_duplicates)} 个事件存在内部重复ID")
            for incident_id, duplicates in internal_duplicates[:5]:  # 只显示前5个
                print(f"  事件 {incident_id}: {duplicates}")
            raise AssertionError(f"事件结构中存在 {len(internal_duplicates)} 个事件有内部重复ID")
        else:
            print("✓ 事件内部ID唯一性验证通过：所有事件内部都没有重复ID")

        print("✓ 所有文件验证通过!")
    except Exception as e:
        print(f"✗ 文件验证失败: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='生成事件对齐评估数据集')
    parser.add_argument('--pure', action='store_true',
                       help='生成纯净版本数据集（只包含有事件对应关系的案例）')

    args = parser.parse_args()

    try:
        extract_standard_event_level_cases_with_annotations(pure_dataset=args.pure)
        dataset_type = "纯净" if args.pure else "标准"
        print(f"\n🎉 {dataset_type}数据集生成成功完成!")
    except Exception as e:
        print(f"\n❌ 数据集生成失败: {e}")
        raise
