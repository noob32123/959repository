from ultralytics import YOLO
YOLO('runs/detect/train/weights/last.pt').val(data='data_A1A9/airplane.yaml')
# YOLO('runs/detect/pretrain/weights/best.pt').val(data='data_A1A9/airplane.yaml')
# YOLO('constrained_best.pt').val(data='data_/airplane.yaml')