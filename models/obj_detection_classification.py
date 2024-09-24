from ultralytics import YOLO
import torch
from pathlib import Path

model = YOLO('yolov8n.pt') 
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

data_path = str(Path('api_out/img2img/data.yaml').resolve())

print(torch.cuda.is_available())
model.train(data=data_path, epochs=30, imgsz=320, batch=32, name='yolov8_custom')