import warnings
import argparse
import pandas as pd
import os
from ultralytics import YOLO

warnings.filterwarnings('ignore')



if __name__ == '__main__':


    train_args = {
        'data': 'data_A1A9/airplane.yaml',
        'imgsz': 640,
        'epochs': 50,
        'batch': 256,
        'workers': 8,
        'device': '0,1,2,3',
        'optimizer': 'SGD',
        'amp': True,
        'cache': True,
        'val': True
    }

    


    model = YOLO('/home/ybc/anaconda3/bin/envs/959/lib/python3.8/site-packages/ultralytics/cfg/models/11/yolo11s.yaml')
    model.load('runs/detect/pretrain/weights/best.pt')
    model.train(**train_args)
    model.val(data='data_A1A9/airplane.yaml')