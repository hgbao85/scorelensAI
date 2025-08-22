import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import csv
import time
import math
from math import atan2, degrees

# -------- CONFIG --------
ANGLE_CHANGE_THRESHOLD = 15    # degrees to detect collision by angle change
IOU_THRESHOLD = 0.0            # any IOU > 0 counts as collision
SPEED_THRESHOLD = 7            # pixels/frame for moving detection in turn counting
COLLISION_SPEED_THRESHOLD = 0  # for collision detection (any movement)
STILL_COOLDOWN = 2             # seconds before considering ball still
BALL_LABELS = ['bi_cam', 'bi_trang', 'bi_vang']
DISPLAY_COLLISION_TIME = 5     # seconds to display collision message on screen

PLAYER_A = 'A'
PLAYER_B = 'B'

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calc_angle(p1, p2):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return degrees(atan2(dy, dx))

def main(video_path, model_path, csv_output_path="carom_event_log.csv", output_video_path="carom_output_demo.mp4"):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cooldown_frames = int(STILL_COOLDOWN * fps)

    # Setup VideoWriter (MP4, H264 codec or use 'mp4v')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    prev_positions = {label: None for label in BALL_LABELS}
    prev_angles = {label: None for label in BALL_LABELS}
    states = {label: "still" for label in BALL_LABELS}
    still_frame_counters = defaultdict(int)

    turn_count = 0
    in_turn = False

    active_collisions = []

    current_player = PLAYER_A
    points = {PLAYER_A: 0, PLAYER_B: 0}

    def assign_balls_for_player(player):
        if player == PLAYER_A:
            return 'bi_trang', {'bi_vang', 'bi_cam'}
        else:
            return 'bi_vang', {'bi_trang', 'bi_cam'}

    cue_ball, object_balls = assign_balls_for_player(current_player)

    hit_object_balls = set()
    cushion_hits = 0
    turn_message = ""

    def reset_turn_state():
        nonlocal hit_object_balls, cushion_hits, turn_message
        hit_object_balls = set()
        cushion_hits = 0
        turn_message = ""

    reset_turn_state()

    with open(csv_output_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Event_Type', 'Turn', 'Frame', 'Timestamp', 'Details'])

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # seconds

            results = model.predict(frame, verbose=False)
            boxes = results[0].boxes
            labels = results[0].names

            current_positions = {label: None for label in BALL_LABELS}
            current_boxes = {label: None for label in BALL_LABELS}

            for box, cls_id in zip(boxes.xyxy, boxes.cls):
                label = labels[int(cls_id)]
                if label in BALL_LABELS:
                    x1, y1, x2, y2 = box
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    current_positions[label] = (cx, cy)
                    current_boxes[label] = (float(x1), float(y1), float(x2), float(y2))

            # --- Turn counting logic ---
            for label in BALL_LABELS:
                if prev_positions[label] is not None and current_positions[label] is not None:
                    speed = distance(prev_positions[label], current_positions[label])

                    if speed > SPEED_THRESHOLD:
                        states[label] = "moving"
                        still_frame_counters[label] = 0
                    else:
                        if states[label] == "moving":
                            still_frame_counters[label] += 1
                            if still_frame_counters[label] >= cooldown_frames:
                                states[label] = "still"
                        else:
                            states[label] = "still"
                else:
                    states[label] = "still"

            if not in_turn and any(s == "moving" for s in states.values()):
                in_turn = True

            # --- Collision detection logic ---
            now = time.time()
            for label in BALL_LABELS:
                if prev_positions[label] and current_positions[label]:
                    speed = distance(prev_positions[label], current_positions[label])
                    if speed > COLLISION_SPEED_THRESHOLD:
                        angle_now = calc_angle(prev_positions[label], current_positions[label])
                        if prev_angles[label] is not None:
                            angle_diff = abs(angle_now - prev_angles[label])
                            angle_diff = angle_diff if angle_diff <= 180 else 360 - angle_diff
                            if angle_diff > ANGLE_CHANGE_THRESHOLD:
                                # Check ball-ball collisions
                                ball_collision_found = False
                                for other_label in BALL_LABELS:
                                    if other_label != label and current_boxes[label] and current_boxes[other_label]:
                                        if iou(current_boxes[label], current_boxes[other_label]) > IOU_THRESHOLD:
                                            pair = tuple(sorted([label, other_label]))
                                            coll_text = f"{pair[0]} with {pair[1]}"
                                            if coll_text not in [c[0] for c in active_collisions]:
                                                active_collisions.append((coll_text, now))
                                                # Log collision to CSV
                                                csv_writer.writerow(['Collision', turn_count, frame_count, f"{timestamp:.2f}", coll_text])
                                            ball_collision_found = True
                                # Cushion collision if no ball collision found
                                if not ball_collision_found:
                                    coll_text = f"{label} with cushion"
                                    if coll_text not in [c[0] for c in active_collisions]:
                                        active_collisions.append((coll_text, now))
                                        csv_writer.writerow(['Collision', turn_count, frame_count, f"{timestamp:.2f}", coll_text])
                        prev_angles[label] = angle_now
                prev_positions[label] = current_positions[label]

            # Remove expired collisions from active list
            active_collisions = [(txt, ts) for txt, ts in active_collisions if now - ts <= DISPLAY_COLLISION_TIME]

            # --- Interpret collisions for scoring ---

            for coll_text, ts in active_collisions:
                if "with cushion" in coll_text:
                    ball = coll_text.split(" with ")[0]
                    if ball == cue_ball:
                        cushion_hits += 1
                else:
                    ball1, ball2 = coll_text.split(" with ")
                    if cue_ball in (ball1, ball2):
                        other_ball = ball2 if ball1 == cue_ball else ball1
                        if other_ball in object_balls:
                            hit_object_balls.add(other_ball)

            # --- Turn change and scoring ---
            if in_turn and all(s == "still" for s in states.values()):
                # Turn ended
                turn_count += 1
                in_turn = False

                extra_turn = False
                if hit_object_balls == object_balls and cushion_hits >= 3:
                    points[current_player] += 1
                    extra_turn = True
                    turn_message = f"Player {current_player} scores +1 and gets extra turn!"
                else:
                    turn_message = f"Player {current_player} turn ends."

                csv_writer.writerow(['Turn_Change', turn_count, frame_count, f"{timestamp:.2f}", f"All balls still - {turn_message}"])

                if not extra_turn:
                    current_player = PLAYER_B if current_player == PLAYER_A else PLAYER_A
                    turn_message += f" Next turn: Player {current_player}"

                cue_ball, object_balls = assign_balls_for_player(current_player)
                reset_turn_state()

            # --- Draw debug info and scoreboard ---
            cv2.putText(frame, f"Turn count: {turn_count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Player A (White) Points: {points[PLAYER_A]}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Player B (Yellow) Points: {points[PLAYER_B]}", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Current turn: Player {current_player}", (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            if turn_message:
                cv2.putText(frame, turn_message, (20, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            for i, label in enumerate(BALL_LABELS):
                cv2.putText(frame, f"{label}: {states[label]}", (20, 220 + i*30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            y_offset = 320
            for coll_text, ts in active_collisions:
                cv2.putText(frame, f"Collision: {coll_text}", (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                y_offset += 25

            out.write(frame)

            # Optional: show on screen as well
            cv2.imshow("Carom Event Logger", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Log final total scores to CSV
        end_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        end_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # seconds
        csv_writer.writerow([
            'FinalScore',
            turn_count,
            end_frame,
            f"{end_timestamp:.2f}",
            f"Player A: {points[PLAYER_A]}, Player B: {points[PLAYER_B]}"
        ])

    # Print final scores to console
    print(f"Final Scores -> Player A: {points[PLAYER_A]}, Player B: {points[PLAYER_B]}")

    cap.release()
    out.release()  # Important: close the video writer!
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "carom_video/0728 (1)(6).mp4"
    model_path = "train15/weights/best.pt"
    main(video_path, model_path)
