import cv2
from detection import AccidentDetectionModel
import numpy as np
import os
from collections import deque, Counter

model = AccidentDetectionModel("model.json", "model_weights.keras")
font = cv2.FONT_HERSHEY_SIMPLEX

FRAME_BUFFER_SIZE = 10
PREDICTION_THRESHOLD = 0.9
FRAME_SKIP = 4  

def startapplication():
    video = cv2.VideoCapture("cars.mp4")
    frame_buffer = deque(maxlen=FRAME_BUFFER_SIZE)
    prediction_buffer = deque(maxlen=FRAME_BUFFER_SIZE)
    frame_id = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame_id += 1

        # Skip frames for speed
        if frame_id % FRAME_SKIP != 0:
            continue

        frame_resized = cv2.resize(frame, (640, 360))

        roi = cv2.resize(frame_resized, (250, 250))
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi_rgb = roi_rgb[np.newaxis, ...]  # Add batch dim

        # Predict
        label, prob = model.predict_accident(roi_rgb)
        confidence = round(prob[0][0], 2)

        if confidence > PREDICTION_THRESHOLD:
            prediction_buffer.append(label)
        else:
            prediction_buffer.append("No Accident")

        # Majority vote
        majority_prediction = Counter(prediction_buffer).most_common(1)[0][0]

        # Show prediction
        if majority_prediction == "Accident":
            avg_conf = np.mean([
                prob[0][0] * 100 for i in range(len(prediction_buffer))
                if prediction_buffer[i] == "Accident"
            ])
            cv2.rectangle(frame_resized, (0, 0), (320, 40), (0, 0, 0), -1)
            cv2.putText(frame_resized, f"Accident {avg_conf:.2f}%", (20, 30), font, 1, (0, 0, 255), 2)

        cv2.imshow("Video", frame_resized)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    startapplication()
