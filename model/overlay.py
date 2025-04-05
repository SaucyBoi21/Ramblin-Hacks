import cv2
import mediapipe as mp
import numpy as np

# Path to your soccer video file
video_path = 'app/Data/morspn6.mp4'
output_path = 'app/processed_data/clip6.mp4'

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

# Open video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file: {video_path}")
    exit()

# Output video setup
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    top_half = frame[:frame_height // 2, :]  # Only top half of the frame

    gray = cv2.cvtColor(top_half, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    lines = cv2.HoughLinesP(dilated_edges, 1, np.pi / 180, threshold=120,
                            minLineLength=120, maxLineGap=5)

    valid_points = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # Only accept near-vertical or near-horizontal lines
            if (85 <= angle <= 95 or angle <= 5):
                # Offset y back to full-frame coordinates
                y1_full = y1
                y2_full = y2
                valid_points.append((x1, y1_full))
                valid_points.append((x2, y2_full))

                # Draw the line (in blue)
                # cv2.line(frame, (x1, y1_full), (x2, y2_full), (255, 0, 0), 2)

    # Draw bounding box if valid lines found
    if valid_points:
        pts = np.array(valid_points)
        x, y, w, h = cv2.boundingRect(pts)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 3)

    # Pose estimation
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=0)
        )

    out.write(frame)
    cv2.imshow("Goal Post Detection (Top Half Only)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved to: {output_path}")
