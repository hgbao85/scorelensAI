import cv2

# --- CONFIG ---
input_path = "C:/Users/ADMIN/OneDrive/Documents/ScoreLen/pool8_video/IMG_6086.mov"
output_path = "cut_clip.mp4"

start_time = 40   # cut start (seconds)
end_time = 70     # cut end (seconds)

# Open video
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError("Error: Cannot open video file.")

# Get properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps

# Clamp times
start_time = max(0, min(start_time, duration))
end_time = max(0, min(end_time, duration))

# Convert to frames
start_frame = int(start_time * fps)
end_frame = int(end_time * fps)

print(f"Video duration: {duration:.2f}s, FPS: {fps}, Total frames: {frame_count}")
print(f"Extracting from {start_time:.2f}s (frame {start_frame}) to {end_time:.2f}s (frame {end_frame})")

# Set the starting frame
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# Define video writer (same size & fps)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Read and save frames until end_frame
current_frame = start_frame
while current_frame < end_frame:
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)
    current_frame += 1

cap.release()
out.release()
print(f"✅ Saved clip from {start_time}s to {end_time}s → {output_path}")
