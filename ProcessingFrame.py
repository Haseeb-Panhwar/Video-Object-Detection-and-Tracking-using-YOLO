import ffmpeg
import cv2
import os
import csv
import torch
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker
from argparse import Namespace
import numpy as np


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou
def process_frames(img_path,model, tracker,output_csv='objectDetection_output_yolov8.csv'):
    csv_file = open(output_csv, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Frame', 'ID', 'Class', 'x1', 'y1', 'x2', 'y2'])

    track_durations = {}
    id_to_class = {}

    results = model(img_path)
    result = results[0]
    dets = []
    names = result.names
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    clss = result.boxes.cls.cpu().numpy().astype(int)  # 👈 cast here!
    print("clss: ", clss, clss.dtype, clss.shape)
    for box, conf, cls_id in zip(boxes, confs, clss):
        x1, y1, x2, y2 = box
        cls_name = model.names[cls_id]
        print([cls_name, int(x1), int(y1), int(x2), int(y2)])
        dets.append([x1, y1, x2, y2, conf, cls_id])

    dets = np.array(dets)
    print(dets.shape)

    dets = torch.tensor(dets)
    img = cv2.imread(img_path)
    img_height, img_width = img.shape[:2]
    img_size = (img_height, img_width)
    img_info = (img_width, img_height)

    online_targets = tracker.update(dets, img_info, img_size)

    next_custom_id = 10000  # Starting ID for reassignments

    for t in online_targets:
        tlwh = t.tlwh
        original_tid = t.track_id

        x1, y1, x2, y2 = tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]

        # Match tracker box to nearest detection to get class
        box_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        min_dist = float('inf')
        best_cls_id = -1

        max_iou = 0
        best_cls_id = -1
        tracker_box = [x1, y1, x2, y2]
        for det in dets:
            det_box = det[:4]
            cls_id = det[-1]
            iou = compute_iou(tracker_box, det_box)
            if iou > max_iou:
                max_iou = iou
                best_cls_id = int(cls_id)

        # Class filtering
        if original_tid not in id_to_class:
            id_to_class[original_tid] = best_cls_id
            tid = original_tid
        else:
            if id_to_class[original_tid] == best_cls_id:
                tid = original_tid
            else:
                # Class conflict: assign a new unique tid
                tid = next_custom_id
                next_custom_id += 1
                id_to_class[tid] = best_cls_id

        # Save to CSV
        csv_writer.writerow([ tid, names[best_cls_id], int(x1), int(y1), int(x2), int(y2)])

        # Track durations


    csv_file.close()

def initialize_tracker():
    tracker_args = Namespace(
        track_thresh=0.5,
        track_buffer=30,
        match_thresh=0.8,
        frame_rate=30,
        aspect_ratio_thresh=1.6,
        min_box_area=10,
        mot20=False
    )
    return BYTETracker(tracker_args, frame_rate=30)

def main():
    # convert_video("car-detection.mkv","car-detection.mp4")
    model = YOLO('yolov8n.pt')
    tracker = initialize_tracker()
    process_frames("frame39.jpg",model, tracker)
if __name__ == "__main__":
    main()