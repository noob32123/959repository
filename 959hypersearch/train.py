import warnings
import argparse
import pandas as pd
import os
from ultralytics import YOLO

warnings.filterwarnings('ignore')


def parse_args():

    parser = argparse.ArgumentParser(description='YOLO训练脚本')

    
    parser.add_argument('--project', type=str, help='项目名称')
    parser.add_argument('--name', type=str, help='实验名称')
    parser.add_argument('--epochs', type=int, help='epoch数')
    parser.add_argument('--lr0', type=float, help='初始学习率')
    parser.add_argument('--lrf', type=float, help='结束学习率')
    parser.add_argument('--dropout', type=float, help='dropout率')
    parser.add_argument('--batch', type=int, help='批次大小')
    parser.add_argument('--optimizer', type=str, help='优化器')
    parser.add_argument('--workers', type=float, help='线程数')
    parser.add_argument('--iou', type=float, help='交并比')
    parser.add_argument('--momentum', type=float, help='动量')
    parser.add_argument('--weight_decay', type=float, help='权重衰减')
    parser.add_argument('--warmup_epochs', type=int, help='热身epoch数')
    parser.add_argument('--warmup_momentum', type=float, help='warmup动量')
    parser.add_argument('--warmup_bias_lr', type=float, help='warmup偏置学习率')
    parser.add_argument('--box', type=float, help='box损失权重')
    parser.add_argument('--cls', type=float, help='分类损失权重')
    parser.add_argument('--dfl', type=float, help='DFL损失权重')
    parser.add_argument('--pose', type=float, help='姿态损失权重')
    parser.add_argument('--kobj', type=float, help='关键点目标损失权重')
    parser.add_argument('--nbs', type=int, help='名义batch size')
    parser.add_argument('--hsv_h', type=float, help='HSV色调增强幅度')
    parser.add_argument('--hsv_s', type=float, help='HSV饱和度增强幅度')
    parser.add_argument('--hsv_v', type=float, help='HSV明度增强幅度')
    parser.add_argument('--degrees', type=float, help='旋转角度范围')
    parser.add_argument('--translate', type=float, help='平移幅度范围')


    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    train_args = {
        'data': 'data_yolo_3073/airplane.yaml',
        'imgsz': 640,
        'epochs': 5,
        'batch': 64,
        'workers': 8,
        'device': '2,3',
        'optimizer': 'Adam',
        'amp': True,
        'cache': True,
        'project': 'test',
        'name': 'experiment_name',
    }


    for arg_name, arg_value in vars(args).items():
        if arg_value is not None:  
            train_args[arg_name] = arg_value
    print(train_args)

    model = YOLO('/home/ybc/anaconda3/bin/envs/959/lib/python3.8/site-packages/ultralytics/cfg/models/11/yolo11s.yaml')
    model.load('yolo11n.pt')
    model.train(**train_args)


    csv_path = train_args['project']+'/'+train_args['name']+'/results.csv'


    try:

        df = pd.read_csv(csv_path)
        

        if 'metrics/mAP50(B)' not in df.columns:
            print(f"错误: CSV 文件中缺少 'metrics/mAP50(B)' 列")
            print(f"可用的列有: {list(df.columns)}")
            exit()
        
        max_map50 = df['metrics/mAP50(B)'].max()
 

        print(f"CSV 文件: {os.path.basename(csv_path)}")
        print(f"MAP50: {max_map50:.4f}")


    except FileNotFoundError:
        print(f"错误: 找不到文件 '{csv_path}'")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")

    

    