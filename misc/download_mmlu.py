#!/usr/bin/env python3
"""
下载 MMLU 数据集到本地，供 lm_eval 离线使用
"""

import os
import json

def download_mmlu_json(output_dir: str = "./dataset/mmlu"):
    """
    下载 MMLU 数据集到本地，保存为 JSON 格式
    
    MMLU 包含 57 个主题，每个主题有 dev/test 两个 split
    """
    from datasets import load_dataset
    
    print("开始下载 MMLU 数据集...")
    print("数据集来源: cais/mmlu")
    print(f"保存到: {os.path.abspath(output_dir)}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # MMLU 的所有子集名称
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
    
    total_samples = 0
    
    # 下载每个子集
    for i, subset in enumerate(subsets):
        subset_dir = os.path.join(output_dir, subset)
        os.makedirs(subset_dir, exist_ok=True)
        
        try:
            print(f"[{i+1}/{len(subsets)}] 下载 {subset}...", end=" ")
            subset_samples = 0
            
            # 下载 dev split
            try:
                ds_dev = load_dataset("cais/mmlu", subset, split="dev")
                dev_file = os.path.join(subset_dir, "dev.json")
                ds_dev.to_json(dev_file)
                subset_samples += len(ds_dev)
            except Exception as e:
                print(f"\n  警告: dev split 下载失败: {e}")
            
            # 下载 test split
            try:
                ds_test = load_dataset("cais/mmlu", subset, split="test")
                test_file = os.path.join(subset_dir, "test.json")
                ds_test.to_json(test_file)
                subset_samples += len(ds_test)
            except Exception as e:
                print(f"\n  警告: test split 下载失败: {e}")
                
            total_samples += subset_samples
            print(f"✓ ({subset_samples} samples)")
            
        except Exception as e:
            print(f"✗ 失败: {e}")
    
    print(f"\n{'='*50}")
    print(f"下载完成!")
    print(f"总样本数: {total_samples}")
    print(f"保存路径: {os.path.abspath(output_dir)}")
    print(f"\n目录结构示例:")
    print(f"  {output_dir}/")
    print(f"    ├── abstract_algebra/")
    print(f"    │   ├── dev.json")
    print(f"    │   └── test.json")
    print(f"    ├── anatomy/")
    print(f"    │   ├── dev.json")
    print(f"    │   └── test.json")
    print(f"    └── ...")
    print(f"{'='*50}")
    
    return output_dir


def verify_mmlu_local(output_dir: str = "./data/mmlu"):
    """
    验证本地 MMLU 数据集是否完整
    """
    import glob
    
    print(f"\n验证本地 MMLU 数据集: {output_dir}")
    
    if not os.path.exists(output_dir):
        print(f"错误: 目录不存在: {output_dir}")
        return False
    
    # 统计文件
    json_files = glob.glob(os.path.join(output_dir, "*", "*.json"))
    print(f"找到 {len(json_files)} 个 JSON 文件")
    
    # 显示前几个文件
    for f in json_files[:5]:
        print(f"  - {f}")
    if len(json_files) > 5:
        print(f"  ... 还有 {len(json_files) - 5} 个文件")
    
    return len(json_files) > 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download MMLU dataset to local")
    parser.add_argument("--output-dir", default="./dataset/mmlu", 
                       help="输出目录 (默认: ./dataset/mmlu)")
    parser.add_argument("--verify", action="store_true",
                       help="验证已下载的数据集")
    args = parser.parse_args()
    
    if args.verify:
        verify_mmlu_local(args.output_dir)
    else:
        output_dir = download_mmlu_json(args.output_dir)
        verify_mmlu_local(output_dir)
