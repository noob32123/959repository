from ultralytics import YOLO
# YOLO('runs/detect/train/weights/last.pt').val(data='data_train/airplane.yaml')
YOLO('constrained_best.pt').val(data='data_train/airplane.yaml')