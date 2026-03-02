#!/usr/bin/env python3
"""
下载 FineWeb-Edu 数据集前 N 条到本地保存
"""

from datasets import load_dataset
import json
import os

def download_and_save_dataset(
    output_dir: str = "./dataset/fineweb_edu",
    num_samples: int = 1000,
    subset: str = "sample-10BT"
):
    """
    下载 FineWeb-Edu 数据集并保存到本地
    
    Args:
        output_dir: 保存目录
        num_samples: 下载样本数
        subset: 数据集子集名称
    """
    print(f"开始下载 FineWeb-Edu (subset={subset}) 前 {num_samples} 条数据...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据集 (先尝试 streaming 模式获取数据)
    try:
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu", 
            subset, 
            split="train",
            streaming=True
        )
        
        # 收集前 N 条数据
        samples = []
        for i, sample in enumerate(dataset):
            if i >= num_samples:
                break
            samples.append(sample)
            if (i + 1) % 100 == 0:
                print(f"已下载: {i + 1}/{num_samples}")
        
        # 保存为 JSON 文件
        output_file = os.path.join(output_dir, f"fineweb_edu_{subset}_{num_samples}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 数据已保存到: {output_file}")
        print(f"✓ 共保存 {len(samples)} 条数据")
        
    except Exception as e:
        print(f"下载失败: {e}")
        print("尝试使用非 streaming 模式...")
        
        # 如果 streaming 失败，尝试非 streaming 模式
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu", 
            subset, 
            split=f"train[:{num_samples}]"
        )
        
        # 保存为 JSON
        output_file = os.path.join(output_dir, f"fineweb_edu_{subset}_{num_samples}.json")
        dataset.to_json(output_file)
        print(f"✓ 数据已保存到: {output_file}")


def load_local_dataset(data_file: str):
    """
    从本地 JSON 文件加载数据集
    
    Args:
        data_file: JSON 文件路径
    
    Returns:
        数据集列表
    """
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    # 下载前 1000 条数据
    download_and_save_dataset(
        output_dir="./dataset/fineweb_edu",
        num_samples=10000,
        subset="sample-10BT"
    )
