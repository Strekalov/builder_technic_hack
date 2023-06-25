import numpy as np
from ultralytics import YOLO

# Load a model
model = YOLO('/home/cv_user/ultralytics/runs/pose/train15/weights/best.pt')  # load a custom model

# Predict with the model
results = model('img013.jpg', show=True, save=True, hide_labels=False,  device=[0,1], half=True, imgsz=2560, save_txt=True)

keypoints = results[0].keypoints  # Masks object
boxes = results[0].boxes
for i, box in enumerate(boxes.cpu().numpy()):

    if box.cls[0] == 1:
        arr = keypoints[i].cpu().numpy()
        print(arr)
        # Последний элемент numpy array является палкой
    elif box.cls[0] == 0:
        arr = keypoints[i].cpu().numpy()
        print(arr)
        # Последний элемент numpy array является ковшом
