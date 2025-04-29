def process_frames2(model, tracker, frame_dir='data', output_csv='objectDetection_output_yolov8.csv'):
    csv_file = open(output_csv, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Frame', 'ID', 'Class', 'x1', 'y1', 'x2', 'y2'])

    frame_files = sorted(os.listdir(frame_dir))
    track_durations = {}
    id_to_class = {}

    for frame_id, image_file in enumerate(frame_files):
        img_path = os.path.join(frame_dir, image_file)
        results = model(img_path)
        nms = model.names
        dets = []
        names = results[0].names
        count = 0

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy()
            clss2 = result.boxes.cls
            print("clss: ", clss, clss.dtype, clss.shape)
            for box, conf, cls_id in zip(boxes, confs, clss):
                x1, y1, x2, y2 = box
                dets.append([x1, y1, x2, y2, conf, cls_id])

                csv_writer.writerow([frame_id, nms[cls_id], int(x1), int(y1), int(x2), int(y2)])


            count+=1
        if len(dets) == 0:
            continue

        dets = torch.tensor(dets)
        img = cv2.imread(img_path)
        img_height, img_width = img.shape[:2]
        img_size = (img_height, img_width)
        img_info = (img_width, img_height)

        online_targets = tracker.update(dets, img_info, img_size)
        print("\nOnline Targets: ",online_targets)
        # for t in online_targets:
        #     tlwh = t.tlwh
        #     tid = t.track_id
        #     cls_id = int(t.score[1]) if isinstance(t.score, tuple) else int(dets[0][-1])
        #     x1, y1, x2, y2 = tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]
        #
        #     csv_writer.writerow([frame_id, tid, names[cls_id], int(x1), int(y1), int(x2), int(y2)])
        #
        #     if tid not in id_to_class:
        #         id_to_class[tid] = names[cls_id]
        #
        #     if tid not in track_durations:
        #         track_durations[tid] = [frame_id, frame_id]
        #     else:
        #         track_durations[tid][1] = frame_id

        # for t in online_targets:
        #     tlwh = t.tlwh
        #     tid = t.track_id
        #     x1, y1, x2, y2 = tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]
        #
        #     # Match the tracker output to the nearest detection (IoU or center distance)
        #     box_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        #     min_dist = float('inf')
        #     best_cls_id = -1
        #
        #     for det in dets:
        #         det_x1, det_y1, det_x2, det_y2, _, cls_id = det
        #         det_center = np.array([(det_x1 + det_x2) / 2, (det_y1 + det_y2) / 2])
        #         dist = np.linalg.norm(box_center - det_center)
        #         if dist < min_dist:
        #             min_dist = dist
        #             best_cls_id = int(cls_id)
        #
        #     # csv_writer.writerow([frame_id, tid, names[best_cls_id], int(x1), int(y1), int(x2), int(y2)])
        #
        #     if tid not in id_to_class:
        #         id_to_class[tid] = names[best_cls_id]
        #
        #     if tid not in track_durations:
        #         track_durations[tid] = [frame_id, frame_id]
        #     else:
        #         track_durations[tid][1] = frame_id

        next_custom_id = 10000  # Starting ID for reassignments

        for t in online_targets:
            tlwh = t.tlwh
            original_tid = t.track_id

            x1, y1, x2, y2 = tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]

            # Match tracker box to nearest detection to get class
            box_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            min_dist = float('inf')
            best_cls_id = -1

            for det in dets:
                det_x1, det_y1, det_x2, det_y2, _, cls_id = det
                det_center = np.array([(det_x1 + det_x2) / 2, (det_y1 + det_y2) / 2])
                dist = np.linalg.norm(box_center - det_center)
                if dist < min_dist:
                    min_dist = dist
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
            # csv_writer.writerow([frame_id, tid, names[best_cls_id], int(x1), int(y1), int(x2), int(y2)])

            # Track durations
            if tid not in track_durations:
                track_durations[tid] = [frame_id, frame_id]
            else:
                track_durations[tid][1] = frame_id

    csv_file.close()
    return track_durations, id_to_class




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