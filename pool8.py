import cv2
import numpy as np
from ultralytics import YOLO
import csv
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
import time

# -------- CONFIG --------
SPEED_THRESHOLD = 3     # pixels/frame for "moving"
NOT_FOUND_TIME = 1      # seconds before "not found"
STILL_COOLDOWN = 1      # seconds of low speed to consider still
np.random.seed(42)

BALL_LABELS = [
    'bi1', 'bi10', 'bi11', 'bi12', 'bi13', 'bi14', 'bi15',
    'bi2', 'bi3', 'bi4', 'bi5', 'bi6', 'bi7', 'bi8', 'bi9', 'bi_trang'
]
BALL_COLORS = {label: tuple(int(c) for c in np.random.randint(0, 255, 3))
               for label in BALL_LABELS}
LABEL_MAP = {i: label for i, label in enumerate(BALL_LABELS)}

# ===== CNN MODEL =====
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (64//8) * (64//8), 128),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)


def classify_ball(cnn_model, device, frame, box, transform):
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img_tensor = transform(crop_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = cnn_model(img_tensor)
        pred_label_idx = outputs.argmax(1).item()
        return LABEL_MAP.get(pred_label_idx, None)


# Helper: map a ball label to a group (solids/stripes) or None for cue/8
def ball_group(label):
    if label == 'bi_trang' or label == 'bi8':
        return None
    try:
        num = int(label.replace('bi', ''))
    except Exception:
        return None
    if 1 <= num <= 7:
        return 'solids'
    if 9 <= num <= 15:
        return 'stripes'
    return None


def group_labels(group):
    if group == 'solids':
        return {f'bi{i}' for i in range(1,8)}
    if group == 'stripes':
        return {f'bi{i}' for i in range(9,16)}
    return set()


def format_list(lst):
    if not lst:
        return 'None'
    return ','.join(lst)


def main(video_path, yolo_model_path, cnn_model_path, csv_output_path="events_pool8.csv", output_video_path="demo_output.mp4"):
    # ===== DEVICE & MODELS =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")



    yolo_model = YOLO(yolo_model_path)
    cnn_model = SimpleCNN(len(BALL_LABELS)).to(device)
    cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=device))
    cnn_model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])

    # ===== VIDEO & TIMING =====
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None
    if fps is None or fps <= 0:
        fps = 30.0  # fallback
    cooldown_frames = int(STILL_COOLDOWN * fps)
    not_found_frames_limit = int(NOT_FOUND_TIME * fps)
    moving_start_frames = int(0.2 * fps)  # 0.2 second above threshold to enter "moving"

    # ===== STATE STRUCTURES (per ball) =====
    prev_positions = {label: None for label in BALL_LABELS}
    current_positions = {label: None for label in BALL_LABELS}
    states = {label: "still" for label in BALL_LABELS}
    speed_vals = {label: 0.0 for label in BALL_LABELS}

    moving_frame_counters = {label: 0 for label in BALL_LABELS}
    still_frame_counters = {label: 0 for label in BALL_LABELS}
    not_found_counters = {label: 0 for label in BALL_LABELS}

    turn_count = 0
    in_turn = False  # based on bi_trang

    # For collision script compatibility
    prev_boxes = {label: None for label in BALL_LABELS}

    # ===== GAME STATE =====
    teams = {
        'A': {'group': None, 'balls_in_hole': set()},
        'B': {'group': None, 'balls_in_hole': set()}
    }
    current_team = 'A'  # who is shooting this turn
    turn_shooter = None

    # Per-turn vars
    turn_first_collision_label = None
    turn_first_collision_group = None
    turn_had_first_collision = False
    turn_fallen_balls = []

    game_over = False

    # ===== VIDEO OUTPUT =====
    # Get video dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    frames_processed = 0
    start_time = time.time()

    with open(csv_output_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Event_Type','Turn','Frame','Timestamp','Details'])

        # progress bar
        pbar = tqdm(total=total_frames, desc="Processing", unit="frame")

        while True:
            ret, frame = cap.read()
            if not ret or game_over:
                break

            # reset per-frame positions
            for label in BALL_LABELS:
                current_positions[label] = None

            frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            # ===== RUN YOLO & CLASSIFY =====
            results = yolo_model.predict(frame, verbose=False)
            boxes = results[0].boxes

            current_boxes = {label: None for label in BALL_LABELS}

            for box, cls_id in zip(boxes.xyxy, boxes.cls):
                # YOLO must output a generic "ball" class, then we refine with CNN
                if results[0].names[int(cls_id)] == "ball":
                    label = classify_ball(cnn_model, device, frame, box, transform)
                    if label is not None and label in BALL_LABELS:
                        x1, y1, x2, y2 = map(int, box)
                        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        current_positions[label] = (cx, cy)
                        current_boxes[label] = (x1, y1, x2, y2)

                        # Draw detection box in the ball's color
                        cv2.rectangle(frame, (x1, y1), (x2, y2),
                                      BALL_COLORS[label], 2)

                        # Keep prev_boxes for collision script usage
                        prev_boxes[label] = (x1, y1, x2, y2)

            # ===== UPDATE STATES (per ball) =====
            for label in BALL_LABELS:
                speed = 0.0
                if current_positions[label] is not None:
                    # visible this frame
                    not_found_counters[label] = 0
                    if prev_positions[label] is not None:
                        speed = distance(prev_positions[label], current_positions[label])
                        speed_vals[label] = speed

                        if speed > SPEED_THRESHOLD:
                            # count frames above threshold before entering "moving"
                            moving_frame_counters[label] += 1
                            if moving_frame_counters[label] >= moving_start_frames:
                                states[label] = "moving"
                                still_frame_counters[label] = 0
                        else:
                            # below threshold
                            moving_frame_counters[label] = 0
                            if states[label] == "moving":
                                still_frame_counters[label] += 1
                                if still_frame_counters[label] >= cooldown_frames:
                                    states[label] = "still"
                            else:
                                states[label] = "still"
                    else:
                        # first time seen -> treat as still
                        speed_vals[label] = 0.0
                        states[label] = "still"

                    prev_positions[label] = current_positions[label]
                else:
                    # not visible
                    speed_vals[label] = 0.0
                    moving_frame_counters[label] = 0
                    not_found_counters[label] += 1
                    if not_found_counters[label] > not_found_frames_limit:
                        if states[label] != "not found":
                            # Log only when it *first* becomes "not found"
                            csv_writer.writerow(
                                ["Not Found", turn_count, frame_count, f"{timestamp:.2f}", f"{label} not found"]
                            )

                            # --- GAME LOGIC: handle ball falling into hole ---
                            # Determine which team is responsible for this turn
                            pocketing_team = turn_shooter if turn_shooter is not None else current_team
                            g = ball_group(label)

                            # Record the fallen ball for game bookkeeping
                            if label not in teams['A']['balls_in_hole'] and label not in teams['B']['balls_in_hole']:
                                if g is not None:
                                    # If groups haven't been assigned yet, assign based on first pocket
                                    if teams['A']['group'] is None and teams['B']['group'] is None:
                                        teams[pocketing_team]['group'] = g
                                        other = 'B' if pocketing_team == 'A' else 'A'
                                        teams[other]['group'] = 'stripes' if g == 'solids' else 'solids'
                                        csv_writer.writerow(["GroupAssigned", turn_count, frame_count, f"{timestamp:.2f}", f"Team {pocketing_team} -> {g}"])

                                    # Add ball to the correct team's "balls_in_hole"
                                    # If groups are defined we can decide which team the ball belongs to
                                    if teams['A']['group'] is not None and teams['B']['group'] is not None:
                                        if g == teams['A']['group']:
                                            teams['A']['balls_in_hole'].add(label)
                                        elif g == teams['B']['group']:
                                            teams['B']['balls_in_hole'].add(label)
                                    else:
                                        # If for some reason groups still not set, assign to pocketing team
                                        teams[pocketing_team]['balls_in_hole'].add(label)

                                else:
                                    # bi_trang or bi8 fell
                                    if label == 'bi_trang':
                                        csv_writer.writerow(["CuePocketed", turn_count, frame_count, f"{timestamp:.2f}", f"Cue ball pocketed by Team {pocketing_team}"])
                                    if label == 'bi8':
                                        # Who pocketed 8-ball? consider turn_shooter
                                        csv_writer.writerow(["EightPocketed", turn_count, frame_count, f"{timestamp:.2f}", f"Team {pocketing_team} pocketed 8-ball"])

                                        # Evaluate game end conditions immediately
                                        pocket_team = pocketing_team
                                        other = 'B' if pocket_team == 'A' else 'A'

                                        pocket_team_group = teams[pocket_team]['group']
                                        all_objects_in_hole = False
                                        if pocket_team_group is not None:
                                            needed = group_labels(pocket_team_group)
                                            # Check if all group's labels (except 8) are in pocket team's holes
                                            if needed.issubset(teams[pocket_team]['balls_in_hole']):
                                                all_objects_in_hole = True

                                        if states['bi_trang'] == 'still' and all_objects_in_hole:
                                            csv_writer.writerow(["GameEnd", turn_count, frame_count, f"{timestamp:.2f}", f"Team {pocket_team} WINS (8-ball pocketed after all objects) "])
                                            game_over = True
                                        else:
                                            csv_writer.writerow(["GameEnd", turn_count, frame_count, f"{timestamp:.2f}", f"Team {other} WINS (invalid 8-ball) "])
                                            game_over = True

                            # record fallen ball in this turn's list (used to decide extra turn)
                            turn_fallen_balls.append(label)

                        states[label] = "not found"

            # ===== COLLISION DETECTION (moved earlier so first-collision is captured while in_turn) =====
            collisions = []
            cue = "bi_trang"
            if current_boxes[cue] is not None:
                x1, y1, x2, y2 = current_boxes[cue]
                expanded_cue_box = (x1-7, y1-7, x2+7, y2+7)

                for label in BALL_LABELS:
                    if label == cue or current_boxes[label] is None:
                        continue
                    if iou(current_boxes[label], expanded_cue_box) > 0:
                        collision_text = f"{label} with {cue}"
                        collisions.append((collision_text, frame_count, label))
                        csv_writer.writerow(["Collision", turn_count, frame_count, f"{timestamp:.2f}", collision_text])

            # If we haven't recorded the first collision this turn, record it from collisions list
            if in_turn and not turn_had_first_collision and len(collisions) > 0:
                # take the first collision this turn
                first_coll = collisions[0]
                _, coll_frame, coll_label = first_coll
                turn_first_collision_label = coll_label
                turn_first_collision_group = ball_group(coll_label)
                turn_had_first_collision = True
                csv_writer.writerow(["FirstCollision", turn_count + 1, coll_frame, f"{timestamp:.2f}", f"{coll_label} hit by bi_trang"])

            # ===== TURN DETECTION (based on cue ball bi_trang) =====
            # When a new moving sequence starts -> new turn begins
            if not in_turn and states.get("bi_trang") == "moving":
                in_turn = True
                turn_shooter = current_team
                # reset per-turn trackers
                turn_first_collision_label = None
                turn_first_collision_group = None
                turn_had_first_collision = False
                turn_fallen_balls = []
                csv_writer.writerow(["TurnStart", turn_count + 1, frame_count, f"{timestamp:.2f}", f"Team {turn_shooter} starts turn"])

            # When all balls are still or not found -> turn ended
            if in_turn and all(state in ("still", "not found") for state in states.values()):
                # Evaluate turn result and decide extra turn / switch
                extra_turn = False
                shooter = turn_shooter if turn_shooter is not None else current_team
                shooter_group = teams[shooter]['group']

                # If first collision happened, check if it was with shooter's group
                if turn_first_collision_group is not None and shooter_group is not None:
                    if turn_first_collision_group == shooter_group:
                        # check if at least one of shooter's group balls fell in this turn
                        if any((ball in group_labels(shooter_group)) for ball in turn_fallen_balls):
                            extra_turn = True

                # Special rule: if groups were unassigned and shooter pocketed any object, they get the extra turn
                if teams['A']['group'] is None and teams['B']['group'] is None:
                    if any(ball for ball in turn_fallen_balls if ball_group(ball) is not None):
                        extra_turn = True

                if extra_turn:
                    csv_writer.writerow(["ExtraTurn", turn_count + 1, frame_count, f"{timestamp:.2f}", f"Team {shooter} granted extra turn"])
                    # keep current_team the same
                else:
                    # switch turn
                    prev_team = current_team
                    current_team = 'B' if current_team == 'A' else 'A'
                    csv_writer.writerow(["TurnSwitch", turn_count + 1, frame_count, f"{timestamp:.2f}", f"Team {prev_team} -> Team {current_team}"])

                # finalize this turn
                turn_count += 1
                in_turn = False
                turn_shooter = None
                # log turn end
                csv_writer.writerow(["TurnEnd", turn_count, frame_count, f"{timestamp:.2f}", "All balls stopped / turn end"])

            # ===== OVERLAY =====
            cv2.putText(frame, f"Turn count: {turn_count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Sidebar states for all balls
            y0 = 80
            for idx, label in enumerate(BALL_LABELS):
                state = states[label]
                if state == "moving":
                    text = f"{label}: moving ({speed_vals[label]:.1f} px/f)"
                else:
                    text = f"{label}: {state}"
                cv2.putText(frame, text, (20, y0 + idx * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, BALL_COLORS[label], 2)

            # show last few collisions on-screen (if any)
            y_coll = y0 + len(BALL_LABELS)*20 + 20
            for coll_text, coll_frame, *_ in collisions[-5:]:
                cv2.putText(frame, coll_text, (20, y_coll),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)
                y_coll += 20

            # ===== NEW: Game info overlay (top-right) =====
            info_x = max(20, frame.shape[1] - 500)
            info_y = 40
            # Current team whose turn it is
            cv2.putText(frame, f"Current Team: {current_team}", (info_x, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            info_y += 30

            # Team's object ball group (TBD if None)
            team_group = teams[current_team]['group'] if teams[current_team]['group'] is not None else 'TBD'
            cv2.putText(frame, f"Object group: {team_group}", (info_x, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            info_y += 26

            # First collision of the current turn
            first_coll_display = turn_first_collision_label if turn_first_collision_label is not None else 'None'
            cv2.putText(frame, f"First collision: {first_coll_display}", (info_x, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            info_y += 26

            # Balls not_found this turn (turn_fallen_balls)
            not_found_display = format_list(turn_fallen_balls)
            cv2.putText(frame, f"Pocketed this turn: {not_found_display}", (info_x, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            info_y += 26

            # Also show both teams' groups and counts for quick reference
            cv2.putText(frame, f"A: {teams['A']['group'] or 'TBD'} ({len(teams['A']['balls_in_hole'])})", (info_x, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
            info_y += 22
            cv2.putText(frame, f"B: {teams['B']['group'] or 'TBD'} ({len(teams['B']['balls_in_hole'])})", (info_x, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

            # write frame to demo video instead of showing it on-screen
            if writer is not None:
                writer.write(frame)

            # update progress bar and fps
            frames_processed += 1
            elapsed = time.time() - start_time
            fps_avg = frames_processed / elapsed if elapsed > 0 else 0.0
            pbar.update(1)
            pbar.set_postfix({'fps': f"{fps_avg:.2f}"})

        pbar.close()

    # cleanup
    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "pool8_video/IMG_6086.MOV"
    yolo_model_path = "train18/weights/best.pt"
    cnn_model_path = "best_cnn_model.pth"
    main(video_path, yolo_model_path, cnn_model_path, "events_pool8_game.csv", "demo_output.mp4")