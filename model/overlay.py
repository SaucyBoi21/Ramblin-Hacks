import cv2
import mediapipe as mp
import numpy as np

# Path to your soccer video file
video_path = 'model/data/videos/3.mp4'
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

# Calculate the height and width for the middle third of the video
upper_third_height = int(frame_height / 3)
middle_third_width_start = int(frame_width / 3)
middle_third_width_end = int(2 * frame_width / 3)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Crop the frame to focus on the middle third (vertically and horizontally)
    cropped_frame = frame[:upper_third_height, middle_third_width_start:middle_third_width_end]

    # Convert cropped frame to grayscale
    gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

    top_half = frame[:frame_height // 2, :]  # Only top half of the frame

    gray = cv2.cvtColor(top_half, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # Hough Line Detection
    lines = cv2.HoughLinesP(dilated_edges, 1, np.pi / 180, threshold=200,
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
        cv2.rectangle(frame, (x + middle_third_width_start, y), (x + middle_third_width_start + w, y + h), (255, 0, 0), 3)

    # Show the frame
    cv2.imshow("Goal Post Detection", frame)

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
