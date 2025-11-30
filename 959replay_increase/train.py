import warnings
import requests
from ultralytics import YOLO
warnings.filterwarnings('ignore')

flask_url = "http://127.0.0.1:5000/increase_mix"
response = requests.get(flask_url)

if __name__ == '__main__':
    model = YOLO('/home/ybc/anaconda3/bin/envs/959/lib/python3.8/site-packages/ultralytics/cfg/models/11/yolo11s.yaml')   # 修改yaml
    model.load('yolo11n.pt')
    model.train(data='data_train/airplane.yaml',
                imgsz=640,
                epochs=50,
                batch=16,
                workers=8,
                device='0',
                optimizer='SGD',
                amp=True,
                cache=True
                )
