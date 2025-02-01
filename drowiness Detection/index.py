from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import playsound
import os
import speech_recognition as sr
from deepface import DeepFace
import random

# Sound alarm function
def sound_alarm(path):
    global alarm_status, alarm_status2, saying
    while alarm_status or alarm_status2:
        playsound.playsound(path)
        if not alarm_status2:
            break
        saying = False

# Eye aspect ratio calculation
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# EAR calculation for both eyes
def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
    return ear, leftEye, rightEye

# Lip distance calculation for yawn detection
def lip_distance(shape):
    top_lip = np.concatenate((shape[50:53], shape[61:64]))
    low_lip = np.concatenate((shape[56:59], shape[65:68]))
    return abs(np.mean(top_lip, axis=0)[1] - np.mean(low_lip, axis=0)[1])

# Voice command recognition
def voice_command_listener():
    global alarm_status, alarm_status2
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    while True:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
            try:
                command = recognizer.recognize_google(audio).lower()
                if "stop" in command:
                    alarm_status = False
                    alarm_status2 = False
                elif "start detection" in command:
                    alarm_status = False
            except sr.UnknownValueError:
                continue

# Argument parsing
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0, help="Index of webcam on system")
ap.add_argument("-a", "--alarm", type=str, default="alert.mp3", help="Alarm sound file path")
args = vars(ap.parse_args())

# Constants
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 20
alarm_status = False
alarm_status2 = False
saying = False
COUNTER = 0

# Load detectors
print("-> Loading the predictor and detector...")
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Start video stream
print("-> Starting Video Stream")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

# Start voice command listener
Thread(target=voice_command_listener, daemon=True).start()

# Centered window with custom size
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Frame", 800, 600)  # Custom size
screen_res = 1920, 1080  # Screen resolution
scale_width = screen_res[0] / 800
scale_height = screen_res[1] / 600
mid_x = int((screen_res[0] - 800) / 2)
mid_y = int((screen_res[1] - 600) / 2)
cv2.moveWindow("Frame", mid_x, mid_y)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Simulate speed detection
    speed = random.randint(0, 120)  # Simulated speed in km/h

    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in rects:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box with thickness 2
        # Optionally, draw a label for the detected face
        cv2.putText(frame, "Face Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Process the facial landmarks and other detections as usual
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        ear, leftEye, rightEye = final_ear(shape)
        distance = lip_distance(shape)

        # Draw eye and lip contours
        for eye in [leftEye, rightEye]:
            cv2.drawContours(frame, [cv2.convexHull(eye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(shape[48:60])], -1, (0, 255, 0), 1)

        # Emotion Detection
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            emotion_analysis = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)
            emotion = emotion_analysis[0]['dominant_emotion']
        except Exception as e:
            emotion = "Unknown"
            print(f"Emotion detection error: {e}")
            

        # Drowsiness detection
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES and not alarm_status:
                alarm_status = True
                Thread(target=sound_alarm, args=(args["alarm"],), daemon=True).start()
            cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0
            alarm_status = False

        # Yawning detection
        if distance > YAWN_THRESH and not alarm_status2 and not saying:
            alarm_status2 = True
            saying = True
            Thread(target=sound_alarm, args=(args["alarm"],), daemon=True).start()
            cv2.putText(frame, "YAWN ALERT!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            alarm_status2 = False

        # Display metrics
        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(frame, f"YAWN: {distance:.2f}", (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Speed: {speed} m/s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Emotion Display (Left-aligned and color changes based on emotions)
        emotion_text = f"Emotion: {emotion}"
        emotion_colors = {
            "happy": (0, 255, 0),  # Green
            "sad": (255, 0, 0),    # Blue
            "angry": (0, 0, 255),  # Red
            "surprise": (0, 255, 255),  # Yellow
            "fear": (255, 165, 0),  # Orange
            "disgust": (128, 0, 128),  # Purple
            "neutral": (255, 255, 255)  # White
        }
        color = emotion_colors.get(emotion, (255, 255, 255))  # Default to white if emotion not found
        cv2.putText(frame, emotion_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
