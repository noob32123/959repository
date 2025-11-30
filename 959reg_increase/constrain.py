import warnings
import requests
import torch
from ultralytics import YOLO
from copy import deepcopy
import os
warnings.filterwarnings('ignore')



if __name__ == '__main__':

    


    pretrained_model = YOLO('runs/detect/pretrain/weights/best.pt')
    

    trained_model = YOLO('runs/detect/train/weights/last.pt')  # 假设best.pt在当前目录
    
    # 保存预训练模型的参数
    initial_params = deepcopy(pretrained_model.model.state_dict())
    
    # 获取训练后模型的参数
    current_params = trained_model.model.state_dict()
    
    lambda_reg = 0.2  # 正则化强度

    print("进行参数约束...")
    
    # 对训练好的模型参数进行约束
    constrained_params = deepcopy(current_params)
    constrained_count = 0
    skipped_count = 0
    
    for name, current_param in current_params.items():
        if name in initial_params:
            initial_param = initial_params[name]
            
            # 检查参数形状是否匹配
            if current_param.shape == initial_param.shape:
                # 对匹配的参数进行约束：向预训练值拉回
                constrained_param = (1 - lambda_reg) * current_param + lambda_reg * initial_param
                constrained_params[name] = constrained_param
                constrained_count += 1
                print(f"✓ 约束参数: {name}, 形状: {current_param.shape}")
            else:
                # 形状不匹配，保持训练后的值
                skipped_count += 1
                print(f"✗ 跳过参数: {name}, 形状不匹配: {current_param.shape} vs {initial_param.shape}")
        else:
            # 预训练模型中不存在的参数，保持训练后的值
            skipped_count += 1
            print(f"✗ 跳过参数: {name}, 预训练模型中不存在")
    
    print(f"\n参数约束完成！")
    print(f"约束参数数量: {constrained_count}")
    print(f"跳过参数数量: {skipped_count}")
    
    # 将约束后的参数加载回模型
    trained_model.model.load_state_dict(constrained_params, strict=False)
    
    # 保存约束后的模型
    output_path = 'constrained_best.pt'
    trained_model.save(output_path)
    print(f"\n约束后的模型已保存为: {output_path}")
    
    # 验证模型是否可以正常加载和使用
    print("验证模型加载...")
    test_model = YOLO(output_path)
    print("模型验证成功！")