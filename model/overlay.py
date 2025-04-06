import cv2
import mediapipe as mp
import numpy as np

for i in range(1,101):

    video_path = f'model/data/videos/{i}.mp4'
    output_path = f'app/processed_data/clip{i}_skeleton_hough_bw.mp4'

    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False,
                        model_complexity=1,
                        enable_segmentation=False,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    # Initialize MediaPipe
    mp_drawing = mp.solutions.drawing_utils

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        exit()


    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')


    out_combined_bw = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Calculate the height and width for the middle third of the video
    upper_third_height = int(frame_height / 3)
    middle_third_width_start = int(frame_width / 3)
    middle_third_width_end = int(2 * frame_width / 3)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break


        # Adjust video
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
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                # Filter lines
                if (85 <= angle <= 95 or angle <= 5):
                    valid_points.append((x1, y1))
                    valid_points.append((x2, y2))

        combined_frame = np.zeros_like(frame)  # Blank black frame for both skeleton and Hough lines
        if valid_points:
            pts = np.array(valid_points)
            x, y, w, h = cv2.boundingRect(pts)
            cv2.rectangle(combined_frame, (x + middle_third_width_start, y),
                        (x + middle_third_width_start + w, y + h), (255, 255, 255), 3)


        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                combined_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),  # White skeleton
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=0)
            )


        out_combined_bw.write(combined_frame)

        cv2.imshow('Skeleton and Hough Frame (Black & White)', combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out_combined_bw.release()
    cv2.destroyAllWindows()

    print(f"Processed black and white combined video saved to: {output_path}")
