import ffmpeg
import cv2
import os
import csv
from ultralytics import YOLO


def convert_video(input_path, output_path):
    ffmpeg.input(input_path).output(output_path).run()

def extract_frames(video_path, output_dir):
    cam = cv2.VideoCapture(video_path)
    os.makedirs(output_dir, exist_ok=True)
    currentframe = 0

    while True:
        ret, frame = cam.read()
        if ret:
            name = os.path.join(output_dir, f'frame{currentframe}.jpg')
            print(f'Creating... {name}')
            cv2.imwrite(name, frame)
            currentframe += 1
        else:
            break

    cam.release()
    cv2.destroyAllWindows()





def process_frames_with_yolo_tracker(frame_dir, model, output_csv='objectDetection_output.csv'):
    csv_file = open(output_csv, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Frame', 'ID', 'Class', 'x1', 'y1', 'x2', 'y2','Confidence Score'])
    frame_files = sorted(os.listdir(frame_dir))
    track_durations = {}
    id_to_class = {}
    for frame_id, image_file in enumerate(frame_files):
        img_path = os.path.join(frame_dir, image_file)

        results = model.track(source=img_path, persist=True, verbose=False,tracker='bytetrack.yaml')
        if not results:
            continue

        result = results[0]
        if result.boxes is None or result.boxes.xyxy is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else [-1] * len(boxes)
        clss = result.boxes.cls.cpu().numpy().astype(int)
        scores = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else [0.0] * len(boxes)

        for box, cls_id, track_id, conf in zip(boxes, clss, ids, scores):
            if conf < 0.6:
                continue

            x1, y1, x2, y2 = map(int, box)
            class_name = model.names[cls_id]
            csv_writer.writerow([frame_id, track_id, class_name, x1, y1, x2, y2, conf])

            if track_id not in id_to_class:
                id_to_class[track_id] = class_name

            # Update track durations
            if track_id not in track_durations:
                track_durations[track_id] = [frame_id, frame_id]
            else:
                track_durations[track_id][1] = frame_id

    csv_file.close()
    return track_durations, id_to_class

def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def save_tracking_durations_with_time(track_durations, id_to_class, video_path, output_csv='tracking_output_time.csv'):
    fps = get_video_fps(video_path)
    if fps <= 0:
        raise ValueError("Invalid FPS from video. Make sure the video path is correct.")

    print("################################################\nFPS of the video: ",fps)
    with open(output_csv, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['ID', 'Class', 'Start Time (s)', 'End Time (s)', 'Duration (s)'])

        for tid, (start_frame, end_frame) in track_durations.items():
            start_time = round(start_frame / fps, 2)
            end_time = round(end_frame / fps, 2)
            duration = round(end_time - start_time, 2)

            print(f"ID {tid} [{id_to_class[tid]}] was tracked from {start_time}s to {end_time}s ({duration}s)")
            csv_writer.writerow([tid, id_to_class[tid], start_time, end_time, duration])


def save_tracking_durations(track_durations, id_to_class, output_csv='tracking_output.csv'):

    print("Track Durations: ",track_durations )
    with open(output_csv, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['ID', 'Class', 'Start Frame', 'End Frame', 'Total Frames'])
        for tid, (start, end) in track_durations.items():
            total = end - start + 1
            print(f'ID {tid} was tracked from frame {start} to {end} ({total} frames)')
            csv_writer.writerow([tid, id_to_class[tid], start, end, total])

def main():
    video_path = "car-detection.mkv"
    convert_video(video_path,"car-detection.mp4")
    extract_frames("car-detection.mp4","data")
    model = YOLO('yolo11x.pt')
    track_durations, id_to_class = process_frames_with_yolo_tracker('data',model)
    save_tracking_durations(track_durations, id_to_class)
    save_tracking_durations_with_time(track_durations, id_to_class, video_path=video_path)
if __name__ == "__main__":
    main()
