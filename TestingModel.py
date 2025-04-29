from ultralytics import YOLO
import os

model = YOLO("yolov8n.pt")
names = model.names
frame_files = sorted(os.listdir('data'))

for frame_id, image_file in enumerate(frame_files):
    img_path = os.path.join('data', image_file)
    results = model(img_path)

    for r in results:
        for c in r.boxes.cls:
            print(names[int(c)])