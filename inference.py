import cv2
import torch
import urllib.request
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from torchvision import transforms

from model import SignEfficientNet
from config import DEVICE, SAVE_PATH, IMAGE_SIZE, LABELS

########################################
# Hand connection pairs for drawing
# (21 landmarks, same as mp.solutions)
########################################

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # Index
    (0, 9), (9, 10), (10, 11), (11, 12),    # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17),              # Palm
]

########################################
# Config
########################################

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = SAVE_PATH

########################################
# Download MediaPipe hand landmark model
# (only needed once)
########################################

HAND_MODEL_PATH = "hand_landmarker.task"

urllib.request.urlretrieve(
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    HAND_MODEL_PATH
)

########################################
# Load Sign Language model
########################################

model = SignEfficientNet(num_classes=24)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.to(DEVICE)
model.eval()

########################################
# Transform
########################################

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),   # convert to true gray first
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),   # replicate to 3ch for EfficientNet
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

########################################
# MediaPipe Hand Landmarker (Tasks API)
########################################

base_options = mp_python.BaseOptions(
    model_asset_path=HAND_MODEL_PATH
)

hand_options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = mp_vision.HandLandmarker.create_from_options(hand_options)

########################################
# Webcam
########################################

cap = cv2.VideoCapture(0)
frame_index = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_index += 1

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Tasks API requires an mp.Image object
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # detect_for_video requires a timestamp in milliseconds
    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

    results = landmarker.detect_for_video(mp_image, timestamp_ms)

    prediction_text = ""

    ####################################
    # Hand detection
    ####################################

    if results.hand_landmarks:

        h, w, _ = frame.shape

        for hand_landmarks in results.hand_landmarks:

            ################################
            # Bounding box
            ################################

            x_coords = [int(lm.x * w) for lm in hand_landmarks]
            y_coords = [int(lm.y * h) for lm in hand_landmarks]

            x_min = max(min(x_coords) - 30, 0)
            y_min = max(min(y_coords) - 30, 0)

            x_max = min(max(x_coords) + 30, w)
            y_max = min(max(y_coords) + 30, h)

            ################################
            # Crop hand
            ################################

            hand_crop = frame[y_min:y_max, x_min:x_max]
            hand_crop = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)  
            hand_crop = cv2.cvtColor(hand_crop, cv2.COLOR_GRAY2RGB)  

            if hand_crop.size != 0:

                ################################
                # Preprocess
                ################################

                image = transform(hand_crop)
                image = image.unsqueeze(0).to(DEVICE)

                ################################
                # Prediction
                ################################

                with torch.no_grad():

                    output = model(image)

                    probs = torch.softmax(output, dim=1)

                    confidence, pred = torch.max(probs, dim=1)

                confidence = confidence.item()
                pred = pred.item()

                letter = LABELS[pred]

                prediction_text = f"{letter} ({confidence:.2f})"

            ################################
            # Draw bounding box
            ################################

            cv2.rectangle(
                frame,
                (x_min, y_min),
                (x_max, y_max),
                (0, 255, 0),
                2
            )

            ################################
            # Draw landmarks with OpenCV
            ################################

            # Convert normalised coords to pixel coords
            pts = [
                (int(lm.x * w), int(lm.y * h))
                for lm in hand_landmarks
            ]

            # Draw connections
            for start, end in HAND_CONNECTIONS:
                cv2.line(
                    frame,
                    pts[start],
                    pts[end],
                    (0, 200, 255),
                    2
                )

            # Draw joint dots
            for x, y in pts:
                cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)
                cv2.circle(frame, (x, y), 4, (0, 150, 255),   1)

    ####################################
    # Show prediction
    ####################################

    cv2.putText(
        frame,
        prediction_text,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Sign Language Detection", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
landmarker.close()
cv2.destroyAllWindows()
