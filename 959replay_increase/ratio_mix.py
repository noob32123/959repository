import os
import shutil
import random
from tqdm import tqdm
from flask import Flask, request, jsonify

app = Flask(__name__)


def create_mixed_dataset_with_val(
        class_dir,
        remain_dir,
        output_dir,
        val_dir,
        mix_ratio,
        seed=42
):
    """
    参数:
        class_dir: 指定类别数据集路径（需包含images/train/, labels/train/）
        remain_dir: 剩余数据集路径
        output_dir: 输出数据集路径
        val_dir: 手动指定的验证集路径（需包含images/val/, labels/val/）
        mix_ratio: 从剩余训练集中抽取的比例
        seed: 随机种子
    """
    random.seed(seed)

    # 创建输出目录结构
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

    # 获取文件列表函数
    def get_file_pairs(data_dir, split='train'):
        label_path = os.path.join(data_dir, 'labels', split)
        if not os.path.exists(label_path):
            return []
        return [
            (f.replace('.txt', '.jpg'), f)
            for f in os.listdir(label_path)
            if f.endswith('.txt')
        ]

    # ===== 处理训练集 =====
    class_train = get_file_pairs(class_dir, 'train')
    remain_train = get_file_pairs(remain_dir, 'train')

    # 从剩余训练集中抽样
    sampled_remain = random.sample(remain_train, int(len(remain_train) * mix_ratio))
    mixed_train = class_train + sampled_remain

    # 复制训练集文件
    for img_file, label_file in tqdm(mixed_train, desc="处理训练集"):
        src_base = class_dir if (img_file, label_file) in class_train else remain_dir
        shutil.copy(
            os.path.join(src_base, 'images', 'train', img_file),
            os.path.join(output_dir, 'images', 'train', img_file)
        )
        shutil.copy(
            os.path.join(src_base, 'labels', 'train', label_file),
            os.path.join(output_dir, 'labels', 'train', label_file)
        )
    shutil.copy2('airplane.yaml', 'data_train/airplane.yaml')
    # ===== 处理验证集 =====
    if val_dir:
        # 获取验证集文件列表
        val_files = get_file_pairs(val_dir, 'val')
    
        # 复制验证集文件
        for img_file, label_file in tqdm(val_files, desc="处理验证集"):
            shutil.copy(
                os.path.join(val_dir, 'images', 'val', img_file),
                os.path.join(output_dir, 'images', 'val', img_file)
            )
            shutil.copy(
                os.path.join(val_dir, 'labels', 'val', label_file),
                os.path.join(output_dir, 'labels', 'val', label_file)
            )

    # 打印统计信息
    print(f"\n{'=' * 40}")
    print(f"训练集统计:")
    print(f"  - 指定类别样本: {len(class_train)}")
    print(f"  - 从剩余集抽取: {len(sampled_remain)} ({mix_ratio * 100:.1f}%)")
    print(f"  - 训练集总数: {len(mixed_train)}")

    if val_dir:
        print(f"\n验证集统计:")
        print(f"  - 验证集样本数: {len(val_files)}")

    print(f"\n输出路径结构:")
    print(f"  - 训练集: {output_dir}/images/train/")
    print(f"  - 验证集: {output_dir}/images/val/" if val_dir else "  - 无验证集")


@app.route('/increase_mix', methods=['GET'])
def increase_mix():
    # 输入路径配置
    CLASS_DIR = "data_increase"  # 指定类别数据集
    REMAIN_DIR = "data_history_buffer"  # 剩余数据集
    VAL_DIR = "data_val"  #手动指定的验证集路径

    # 输出路径
    OUTPUT_DIR = "data_train"

    # 运行参数
    create_mixed_dataset_with_val(
        class_dir=CLASS_DIR,
        remain_dir=REMAIN_DIR,
        output_dir=OUTPUT_DIR,
        val_dir=VAL_DIR,
        mix_ratio=0.5,
        seed=42
    )

    path = OUTPUT_DIR
    response = {
        "path": path,

    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)