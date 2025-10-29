import cv2
from ultralytics import YOLO
import csv

# Keypoint indices for COCO dataset
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Limbs to draw
LIMBS = [
    ('nose', 'left_eye'),
    ('nose', 'right_eye'),
    ('left_eye', 'left_ear'),
    ('right_eye', 'right_ear'),
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_elbow'),
    ('right_shoulder', 'right_elbow'),
    ('left_elbow', 'left_wrist'),
    ('right_elbow', 'right_wrist'),
    ('left_shoulder', 'left_hip'),
    ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'),
    ('left_hip', 'left_knee'),
    ('right_hip', 'right_knee'),
    ('left_knee', 'left_ankle'),
    ('right_knee', 'right_ankle')
]

def main():
    # Load the YOLOv8-pose model
    model = YOLO('yolov8l-pose.pt')

    # Open the video file
    video_path = 'squash_video.mp4'
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('animation.mp4', fourcc, fps, (width, height))

    # Open CSV file for writing keypoints
    with open('keypoints.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header
        header = ['frame_id', 'person_id']
        for keypoint_name in KEYPOINT_DICT.keys():
            header.extend([f'{keypoint_name}_x', f'{keypoint_name}_y'])
        csv_writer.writerow(header)

        frame_id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run pose detection
            results = model(frame, save=False)

            # Draw keypoints and limbs
            for result in results:
                for person_id, person in enumerate(result.keypoints):
                    keypoints = person.xy[0]
                    row = [frame_id, person_id]
                    for i in range(len(KEYPOINT_DICT)):
                        if i < len(keypoints):
                            row.extend([keypoints[i][0].item(), keypoints[i][1].item()])
                        else:
                            row.extend([0, 0])
                    csv_writer.writerow(row)

                    for limb in LIMBS:
                        start_point_name = limb[0]
                        end_point_name = limb[1]

                        start_point_idx = KEYPOINT_DICT[start_point_name]
                        end_point_idx = KEYPOINT_DICT[end_point_name]

                        if len(keypoints) > start_point_idx and len(keypoints) > end_point_idx:
                            x1, y1 = keypoints[start_point_idx]
                            x2, y2 = keypoints[end_point_idx]

                            if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                                cv2.circle(frame, (int(x1), int(y1)), 5, (0, 0, 255), -1)
                                cv2.circle(frame, (int(x2), int(y2)), 5, (0, 0, 255), -1)

            # Write the frame to the output video
            out.write(frame)
            frame_id += 1

    # Release everything when job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
