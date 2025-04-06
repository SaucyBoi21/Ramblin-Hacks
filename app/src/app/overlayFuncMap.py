import cv2
import mediapipe as mp
import numpy as np

def overlay_prediction(video_path, output_path, predicted_position):
    GOAL_WIDTH_M = 7.32
    GOAL_HEIGHT_M = 2.44

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False,
                        model_complexity=1,
                        enable_segmentation=False,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    upper_third_height = int(frame_height / 3)
    middle_third_width_start = int(frame_width / 3)
    middle_third_width_end = int(2 * frame_width / 3)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Hough Line Detection
        cropped_frame = frame[:upper_third_height, middle_third_width_start:middle_third_width_end]
        gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
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
                if (85 <= angle <= 95 or angle <= 5):
                    valid_points.extend([(x1, y1), (x2, y2)])

        # Draw bounding box around Hough lines (goalposts)
        goal_rect = None
        if valid_points:
            pts = np.array(valid_points)
            goal_x, goal_y, goal_w, goal_h = cv2.boundingRect(pts)
            goal_rect = (goal_x, goal_y, goal_w, goal_h)
            cv2.rectangle(frame,
              (goal_x + middle_third_width_start, goal_y),
              (goal_x + middle_third_width_start + goal_w, goal_y + goal_h),
              (255, 0, 0), 4)  # Blue and thicker


        # Draw predicted ball position if valid
        if predicted_position and goal_rect:
            ball_x_m, ball_y_m = predicted_position
            goal_x, goal_y, goal_w, goal_h = goal_rect

            ball_x_px = int(goal_x + (ball_x_m / GOAL_WIDTH_M) * goal_w + middle_third_width_start)
            ball_y_px = int(goal_y + goal_h - (ball_y_m / GOAL_HEIGHT_M) * goal_h)

            cv2.circle(frame, (ball_x_px, ball_y_px), 10, (0, 0, 255), -1)
            cv2.putText(frame, "Predicted Ball", (ball_x_px + 10, ball_y_px - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Pose tracking
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=0)
            )

        out.write(frame)

    cap.release()
    out.release()
    pose.close()
