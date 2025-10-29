import cv2
from ultralytics import YOLO
import csv
import yt_dlp
import os

# --- Configuration ---
YOUTUBE_URL = 'https://www.youtube.com/watch?v=i06Cr5oBi0U'
VIDEO_FILENAME = 'squash_video.mp4'
OUTPUT_VIDEO_FILENAME = 'animation.mp4'
OUTPUT_CSV_FILENAME = 'keypoints.csv'
YOLO_MODEL = 'yolov8l-pose.pt'

# --- Keypoint Definitions (COCO dataset) ---
KEYPOINT_DICT = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
}

LIMBS = [
    ('nose', 'left_eye'), ('nose', 'right_eye'), ('left_eye', 'left_ear'),
    ('right_eye', 'right_ear'), ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_elbow'), ('right_shoulder', 'right_elbow'),
    ('left_elbow', 'left_wrist'), ('right_elbow', 'right_wrist'),
    ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'), ('left_hip', 'left_knee'),
    ('right_hip', 'right_knee'), ('left_knee', 'left_ankle'),
    ('right_knee', 'right_ankle')
]

# --- Main Functions ---

def download_video(url, output_path):
    """Downloads a video from a given URL if it doesn't already exist."""
    if os.path.exists(output_path):
        print(f"Video '{output_path}' already exists. Skipping download.")
        return

    print(f"Downloading video from {url} to '{output_path}'...")
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': output_path,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print("Download complete.")
    except Exception as e:
        print(f"An error occurred during download: {e}")
        print("Please ensure you have a working internet connection and that 'ffmpeg' is installed on your system.")
        exit(1)


def process_video(video_path, model):
    """Processes the video to detect poses, generate an animation, and create a keypoints CSV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return

    # Get video properties for the output file
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_FILENAME, fourcc, fps, (width, height))

    print(f"Processing video and generating '{OUTPUT_VIDEO_FILENAME}' and '{OUTPUT_CSV_FILENAME}'...")

    # Prepare CSV file
    with open(OUTPUT_CSV_FILENAME, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        header = ['frame_id', 'person_id']
        for keypoint_name in KEYPOINT_DICT.keys():
            header.extend([f'{keypoint_name}_x', f'{keypoint_name}_y'])
        csv_writer.writerow(header)

        frame_id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run pose detection on the frame
            results = model(frame, save=False, verbose=False)

            # Process detections
            for result in results:
                for person_id, person_keypoints in enumerate(result.keypoints):
                    keypoints = person_keypoints.xy[0]

                    # Write keypoints to CSV
                    row = [frame_id, person_id]
                    keypoints_cpu = keypoints.cpu().numpy()
                    for i in range(len(KEYPOINT_DICT)):
                        if i < len(keypoints_cpu):
                            row.extend([keypoints_cpu[i][0], keypoints_cpu[i][1]])
                        else:
                            row.extend([0, 0]) # Pad if keypoint is missing
                    csv_writer.writerow(row)

                    # Draw limbs and keypoints on the frame
                    for limb_start, limb_end in LIMBS:
                        start_idx = KEYPOINT_DICT[limb_start]
                        end_idx = KEYPOINT_DICT[limb_end]

                        if start_idx < len(keypoints) and end_idx < len(keypoints):
                            x1, y1 = int(keypoints[start_idx][0]), int(keypoints[start_idx][1])
                            x2, y2 = int(keypoints[end_idx][0]), int(keypoints[end_idx][1])

                            # Draw only if the keypoints are detected
                            if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.circle(frame, (x1, y1), 5, (0, 0, 255), -1)
                                cv2.circle(frame, (x2, y2), 5, (0, 0, 255), -1)

            # Write the annotated frame to the output video
            out.write(frame)
            if frame_id % 10 == 0:
                print(f"  Processed frame {frame_id}...")
            frame_id += 1

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Processing complete.")


def main():
    """Main function to run the entire process."""
    # Step 1: Download the video if it's not already there
    download_video(YOUTUBE_URL, VIDEO_FILENAME)

    # Step 2: Load the YOLO model
    print("Loading YOLOv8 model...")
    try:
        model = YOLO(YOLO_MODEL)
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        print("Please ensure the 'ultralytics' package is installed correctly.")
        exit(1)

    # Step 3: Process the video
    process_video(VIDEO_FILENAME, model)

if __name__ == '__main__':
    main()
