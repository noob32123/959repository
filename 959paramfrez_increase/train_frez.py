import warnings
import torch
import torch.nn as nn
from ultralytics import YOLO
from copy import deepcopy

warnings.filterwarnings('ignore')


class ParameterIsolationTrainer:
    def __init__(self, model_path, ckpt_path):
        # 加载原始模型
        self.model = YOLO(model_path)
        self.model.load(ckpt_path)


        # 保存一份冻结的旧模型（teacher）
        self.old_model = deepcopy(self.model.model).eval()
        for p in self.old_model.parameters():
            p.requires_grad = False



    def get_all_convs(self,module):
        """递归获取模块里的所有 Conv2d 层"""
        convs = []
        if isinstance(module, nn.Conv2d):
            convs.append(module)
        else:
            for child in module.children():
                convs.extend(self.get_all_convs(child))
        return convs


    def extend_for_new_classes(self, new_class_count):
        """自动适配 YOLOv5/v7/v8/v11 Detect 层的扩展函数"""
        detect_layer = self.model.model.model[-1]  # 检测头
        nc_old = detect_layer.nc
        nc_new = nc_old + new_class_count
        detect_layer.nc = nc_new
        detect_layer.no = nc_new + 5  # 每个类别 + box(4) + obj(1)

        # 根据版本选择层
        if hasattr(detect_layer, "m"):  # YOLOv5/v7
            layer_lists = [detect_layer.m]
        elif hasattr(detect_layer, "cv2") and hasattr(detect_layer, "cv3"):  # YOLOv8/YOLO11
            layer_lists = [detect_layer.cv2, detect_layer.cv3]
        else:
            raise AttributeError(f"未知 Detect 结构: {type(detect_layer)}，请检查 Ultralytics 版本")

        # 遍历所有 Conv2d 并扩展
        for layer_list in layer_lists:
            for i, block in enumerate(layer_list):
                convs = self.get_all_convs(block)  # 从 Sequential 中取出 Conv2d
                for conv in convs:
                    old_out, in_ch, k1, k2 = conv.weight.shape
                    has_bias = conv.bias is not None
                    old_bias = conv.bias.data if has_bias else None

                    # 新输出大小
                    na = getattr(detect_layer, "na", 3)  # number of anchors
                    no = detect_layer.no  # number of outputs per anchor
                    new_out = na * no  # total output channels
                    new_conv = nn.Conv2d(in_ch, new_out, k1, k2, bias=has_bias)
                    new_conv.to(conv.weight.device)

                    with torch.no_grad():
                        # Ensure we don't try to copy more weights than we have
                        copy_out = min(old_out, new_out)
                        new_conv.weight[:copy_out] = conv.weight[:copy_out]
                        if has_bias:
                            new_conv.bias[:copy_out] = old_bias[:copy_out]

                    # 替换 Sequential 里的 Conv2d
                    for name, child in block.named_children():
                        if child is conv:
                            setattr(block, name, new_conv)

        print(f"🔧 已扩展 Detect 层: {nc_old} -> {nc_new} 类别")



    def train_new_task(self):
        """只训练新类别参数，旧参数冻结"""
        # 获取检测头层索引
        detect_layer = self.model.model.model[-1]  # 检测头总是最后一层
        detect_idx = len(self.model.model.model) - 1
        
        for name, param in self.model.model.named_parameters():
            print(name, param.requires_grad)

        # 根据不同版本找到对应的检测头参数
        detect_pattern = f"model.{detect_idx}"
        if hasattr(detect_layer, "m"):  # YOLOv5/v7
            detect_pattern += ".m"
        elif hasattr(detect_layer, "cv2"):  # YOLOv8/YOLO11
            detect_pattern += ".cv2"
            
        # 冻结所有参数，只训练检测头
        for name, param in self.model.model.named_parameters():
            param.requires_grad = detect_pattern in name
            
        for name, param in self.model.model.named_parameters():
            print(name, param.requires_grad)
        results = self.model.train(
            imgsz=640,
            data="data_A1A9/airplane.yaml",
            epochs=10,
            batch=256,
            device="0,1,2,3",
            optimizer="SGD",
            amp=True,
            cache=True,
        )
        for name, param in self.model.model.named_parameters():
            print(name, param.requires_grad)
        
        return results


if __name__ == "__main__":
    # 初始化参数隔离训练器
    trainer = ParameterIsolationTrainer(
        model_path="/home/ybc/anaconda3/bin/envs/959/lib/python3.8/site-packages/ultralytics/cfg/models/11/yolo11s.yaml",
        ckpt_path="runs/detect/pretrain/weights/best.pt"
    )


    # 假设新任务有 2 个新类别
    trainer.extend_for_new_classes(new_class_count=2)

    # 增量训练
    results = trainer.train_new_task()

