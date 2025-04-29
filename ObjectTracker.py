import ffmpeg
import cv2
import os
import csv
import torch
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker
from argparse import Namespace


ffmpeg.input("peoplenyc.mkv").output('peoplenyc.mp4').run()

cam = cv2.VideoCapture("peoplenyc.mkv")

try:

    if not os.path.exists('data'):
        os.makedirs('data')

except OSError:
    print('Error: Creating directory of data')

currentframe = 0

while True:

    ret, frame = cam.read()

    if ret:
        name = './data/frame' + str(currentframe) + '.jpg'
        print('Creating...' + name)

        cv2.imwrite(name, frame)

        currentframe += 1
    else:
        break

cam.release()
cv2.destroyAllWindows()


# Load model
model = YOLO('yolov8n.pt')

# Create args as Namespace instead of dict
tracker_args = Namespace(
    track_thresh=0.5,
    track_buffer=30,
    match_thresh=0.8,
    frame_rate=30,
    aspect_ratio_thresh=1.6,
    min_box_area=10,
    mot20=False
)

tracker = BYTETracker(tracker_args, frame_rate=30)

# Output CSV file
output_path = 'objectDetection_output_yolov8.csv'
csv_file = open(output_path, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Frame', 'ID', 'Class', 'x1', 'y1', 'x2', 'y2'])


# Map to keep track of durations
track_durations = {}


# Frame loop
frame_dir = 'data'
frame_files = sorted(os.listdir(frame_dir))
id_to_class = {}

for frame_id, image_file in enumerate(frame_files):
    img_path = os.path.join(frame_dir, image_file)
    results = model(img_path)

    dets = []
    names = results[0].names

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy()

        for box, conf, cls_id in zip(boxes, confs, clss):
            x1, y1, x2, y2 = box
            dets.append([x1, y1, x2, y2, conf, cls_id])

    dets = torch.tensor(dets)
    if len(dets) == 0:
        continue

    # Track with ByteTrack
    img = cv2.imread(img_path)
    img_height, img_width = img.shape[:2]
    img_size = (img_height, img_width)
    img_info = (img_width, img_height)

    # Update tracker
    online_targets = tracker.update(dets, img_info, img_size)

    for t in online_targets:
        tlwh = t.tlwh
        tid = t.track_id
        cls_id = int(t.score[1]) if isinstance(t.score, tuple) else int(dets[0][-1])
        x1, y1, x2, y2 = tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]

        # Write to CSV
        csv_writer.writerow([frame_id, tid, names[cls_id], int(x1), int(y1), int(x2), int(y2)])

        if tid not in id_to_class:
            id_to_class[tid] = names[cls_id]

        # Update duration tracking
        if tid not in track_durations:
            track_durations[tid] = [frame_id, frame_id]
        else:
            track_durations[tid][1] = frame_id

csv_file.close()

output_path = 'tracking_output_yolov8.csv'
csv_file2 = open(output_path, mode='w', newline='')
csv_writer2 = csv.writer(csv_file2)
csv_writer2.writerow(['ID', 'Class', 'Start Frame', 'End Frame', 'Total Frames'])

# Print or save durations
print("Object durations:")
for tid, (start, end) in track_durations.items():
    print(f'ID {tid} was tracked from frame {start} to {end} ({end - start + 1} frames)')
    csv_writer2.writerow([tid, id_to_class[tid],start, end, int(end - start + 1)])