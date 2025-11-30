import warnings
from ultralytics import YOLO
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    model = YOLO('/home/ybc/anaconda3/bin/envs/959/lib/python3.8/site-packages/ultralytics/cfg/models/11/yolo11s.yaml')   # 修改yaml
    model.load('runs/detect/50epoch/weights/best.pt')
    model.train(data='data_train/airplane.yaml',
                imgsz=640,
                epochs=50,
                batch=256,
                workers=8,
                device='0,1,2,3',
                optimizer='SGD',
                amp=True,
                cache=True
                )