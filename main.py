import cv2
import os
import argparse
import mediapipe as mp

def process_img(img, face_detection):
    if img is None:
        return img

    H, W, _ = img.shape

    # Convert image to RGB and process it with the face detection model
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    # Check if any faces are detected
    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            # Convert relative bounding box to absolute coordinates
            x1 = int(bbox.xmin * W)
            y1 = int(bbox.ymin * H)
            w = int(bbox.width * W)
            h = int(bbox.height * H)

            # Ensure the bounding box coordinates are within the image bounds
            x1 = max(0, min(x1, W-1))  # Ensure x1 is within valid range
            y1 = max(0, min(y1, H-1))  # Ensure y1 is within valid range
            x2 = max(0, min(x1 + w, W))  # Ensure the width doesn't exceed bounds
            y2 = max(0, min(y1 + h, H))  # Ensure the height doesn't exceed bounds

            # Only blur if the region has a valid size
            if x2 > x1 and y2 > y1:
                img[y1:y2, x1:x2, :] = cv2.blur(img[y1:y2, x1:x2, :], (30, 30))

    return img

# Argument parsing
parser = argparse.ArgumentParser()

parser.add_argument("--mode", default='webcam', help="Mode to run: 'image', 'video', or 'webcam'")
parser.add_argument("--filePath", default=None, help="Path to the image or video file if mode is not webcam")

args = parser.parse_args()

# Create output directory if it doesn't exist
output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    if args.mode == "image":
        if args.filePath is None:
            print("Please provide a valid file path for image mode.")
        else:
            img = cv2.imread(args.filePath)
            if img is None:
                print(f"Error: Could not read image from {args.filePath}")
            else:
                img = process_img(img, face_detection)
                cv2.imwrite(os.path.join(output_dir, 'output.png'), img)

    elif args.mode == "video":
        if args.filePath is None:
            print("Please provide a valid file path for video mode.")
        else:
            cap = cv2.VideoCapture(args.filePath)
            if not cap.isOpened():
                print(f"Error: Could not open video file {args.filePath}")
            else:
                ret, frame = cap.read()

                if ret:
                    output_video = cv2.VideoWriter(os.path.join(output_dir, 'output.mp4'),
                                                   cv2.VideoWriter_fourcc(*'MP4V'),
                                                   25,
                                                   (frame.shape[1], frame.shape[0]))

                    while ret:
                        frame = process_img(frame, face_detection)
                        output_video.write(frame)
                        ret, frame = cap.read()

                    output_video.release()
                cap.release()

    elif args.mode == "webcam":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not access the webcam.")
        else:
            ret, frame = cap.read()

            while ret:
                # Process frame and handle case where no face is detected
                frame = process_img(frame, face_detection)
                cv2.imshow('frame', frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

                ret, frame = cap.read()

            cap.release()
            cv2.destroyAllWindows()
