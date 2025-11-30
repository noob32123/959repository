import os
import shutil
from ultralytics import YOLO

# 配置参数
MODEL_PATH = "runs/detect/pretrain_halflabeled/weights/best.pt"  # 你的YOLO模型路径
UNLABELED_DIR = "unlabeled/images/train"  # 无标签图片目录
ORIGINAL_DIR = "labeled"  # 原始数据集目录
OUTPUT_DIR = "data_train"  # 输出目录
CONF_THRESH = 0.5  # 置信度阈值

def main():
    # 加载模型
    print("加载YOLO模型...")
    model = YOLO(MODEL_PATH)
    
    # 创建输出目录
    os.makedirs(f"{OUTPUT_DIR}/images/train", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/labels/train", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/images/val", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/labels/val", exist_ok=True)
    
    # 1. 复制原始数据集
    print("复制原始数据集...")
    
    # 复制训练集
    if os.path.exists(f"{ORIGINAL_DIR}/images/train"):
        for img_file in os.listdir(f"{ORIGINAL_DIR}/images/train"):
            if img_file.endswith(('.jpg', '.png', '.jpeg')):
                # 复制图片
                shutil.copy2(f"{ORIGINAL_DIR}/images/train/{img_file}", f"{OUTPUT_DIR}/images/train/{img_file}")
                # 复制标签
                label_file = img_file.rsplit('.', 1)[0] + '.txt'
                if os.path.exists(f"{ORIGINAL_DIR}/labels/train/{label_file}"):
                    shutil.copy2(f"{ORIGINAL_DIR}/labels/train/{label_file}", f"{OUTPUT_DIR}/labels/train/{label_file}")
    
    # 复制验证集
    if os.path.exists(f"{ORIGINAL_DIR}/images/val"):
        for img_file in os.listdir(f"{ORIGINAL_DIR}/images/val"):
            if img_file.endswith(('.jpg', '.png', '.jpeg')):
                # 复制图片
                shutil.copy2(f"{ORIGINAL_DIR}/images/val/{img_file}", f"{OUTPUT_DIR}/images/val/{img_file}")
                # 复制标签
                label_file = img_file.rsplit('.', 1)[0] + '.txt'
                if os.path.exists(f"{ORIGINAL_DIR}/labels/val/{label_file}"):
                    shutil.copy2(f"{ORIGINAL_DIR}/labels/val/{label_file}", f"{OUTPUT_DIR}/labels/val/{label_file}")
    
    # 2. 生成伪标签并复制（全部放到训练集）
    print("生成伪标签...")
    if os.path.exists(UNLABELED_DIR):
        for img_file in os.listdir(UNLABELED_DIR):
            if img_file.endswith(('.jpg', '.png', '.jpeg')):
                img_path = f"{UNLABELED_DIR}/{img_file}"
                
                # 预测
                results = model(img_path, conf=CONF_THRESH, verbose=False)
                
                # 复制图片到训练集
                new_img_name = f"pseudo_{img_file}"
                shutil.copy2(img_path, f"{OUTPUT_DIR}/images/train/{new_img_name}")
                
                # 生成标签到训练集
                label_file = f"pseudo_{img_file.rsplit('.', 1)[0]}.txt"
                label_path = f"{OUTPUT_DIR}/labels/train/{label_file}"
                
                with open(label_path, 'w') as f:
                    for result in results:
                        boxes = result.boxes
                        if boxes is not None:
                            for box in boxes:
                                cls = int(box.cls.item())
                                xywh = box.xywhn[0].cpu().numpy()
                                f.write(f"{cls} {xywh[0]:.6f} {xywh[1]:.6f} {xywh[2]:.6f} {xywh[3]:.6f}\n")
    
    total_train_images = len(os.listdir(f"{OUTPUT_DIR}/images/train"))
    total_train_labels = len(os.listdir(f"{OUTPUT_DIR}/labels/train"))
    total_val_images = len(os.listdir(f"{OUTPUT_DIR}/images/val"))
    total_val_labels = len(os.listdir(f"{OUTPUT_DIR}/labels/val"))
    
    print("完成！")
    print(f"训练集: {total_train_images} 张图片, {total_train_labels} 个标签")
    print(f"验证集: {total_val_images} 张图片, {total_val_labels} 个标签")

if __name__ == "__main__":
    main()