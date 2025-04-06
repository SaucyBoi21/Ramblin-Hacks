import cv2
import os

for i in range(1,59):
    # Paths
    input_path = f'app/processed_data/{i}.mp4'  # Replace with your path
    output_path = f'app/processed_data/{i+58}.mp4'

    # Open the video file
    cap = cv2.VideoCapture(input_path)

    # Check if opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'

    # Output writer
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip horizontally (mirror)
        mirrored_frame = cv2.flip(frame, 1)

        # Write to output
        out.write(mirrored_frame)

    # Release everything
    cap.release()
    out.release()
    print(f"Mirrored video saved to {output_path}")
