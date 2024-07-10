from ultralytics import YOLO
import torch

model = YOLO('yolov8n.pt') 
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

print(torch.cuda.is_available())
model.train(data='./yolo_dataset_test/data.yaml', epochs=20, imgsz=640, batch=4, name='yolov8_custom')