#!/usr/bin/env python3
"""
生成 MMLU 本地任务配置文件
"""

import os
import argparse

def generate_configs(data_dir: str, output_dir: str):
    """
    为所有 MMLU 子集生成本地任务配置文件
    
    Args:
        data_dir: MMLU 本地数据目录，例如 ./dataset/mmlu
        output_dir: 配置文件输出目录
    """
    subsets = [
        "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
        "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
        "college_medicine", "college_physics", "computer_security", "conceptual_physics",
        "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic",
        "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
        "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
        "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
        "high_school_physics", "high_school_psychology", "high_school_statistics",
        "high_school_us_history", "high_school_world_history", "human_aging", "human_sexuality",
        "international_law", "jurisprudence", "logical_fallacies", "machine_learning",
        "management", "marketing", "medical_genetics", "miscellaneous", "moral_disputes",
        "moral_scenarios", "nutrition", "philosophy", "prehistory", "professional_accounting",
        "professional_law", "professional_medicine", "professional_psychology", "public_relations",
        "security_studies", "sociology", "us_foreign_policy", "virology", "world_religions"
    ]
    
    os.makedirs(output_dir, exist_ok=True)
    
    for subset in subsets:
        config = f'''task: mmlu_local_{subset}
dataset_path: json
dataset_kwargs:
  data_files:
    dev: {data_dir}/{subset}/dev.json
    test: {data_dir}/{subset}/test.json
test_split: test
fewshot_split: dev
fewshot_config:
  sampler: first_n
output_type: multiple_choice
doc_to_text: "{{{{question.strip()}}}}\\nA. {{{{choices[0]}}}}\\nB. {{{{choices[1]}}}}\\nC. {{{{choices[2]}}}}\\nD. {{{{choices[3]}}}}\\nAnswer:"
doc_to_choice: ["A", "B", "C", "D"]
doc_to_target: answer
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
'''
        with open(f"{output_dir}/mmlu_local_{subset}.yaml", "w") as f:
            f.write(config)
    
    # 生成一个包含所有任务的 group 配置（支持加权平均）
    group_config = f"""group: mmlu_local
task:
"""
    for subset in subsets:
        group_config += f"  - mmlu_local_{subset}\n"
    
    # 添加 aggregate_metric_list 实现按样本数加权平均
    group_config += """aggregate_metric_list:
  - metric: acc
    aggregation: mean
    weight_by_size: true
"""
    
    with open(f"{output_dir}/_mmlu_local.yaml", "w") as f:
        f.write(group_config)
    
    print(f"✓ 已生成 {len(subsets)} 个任务配置文件到: {output_dir}")
    print(f"✓ Group 配置: {output_dir}/_mmlu_local.yaml")
    print(f"\n使用方式:")
    print(f"  lm_eval --include_path {output_dir} --tasks mmlu_local")
    print(f"  # 或运行单个任务")
    print(f"  lm_eval --include_path {output_dir} --tasks mmlu_local_abstract_algebra")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MMLU local task configs")
    parser.add_argument("--data-dir", default="./dataset/mmlu",
                       help="MMLU 本地数据目录")
    parser.add_argument("--output-dir", default="./dataset/mmlu_local",
                       help="配置文件输出目录")
    args = parser.parse_args()
    
    generate_configs(args.data_dir, args.output_dir)
