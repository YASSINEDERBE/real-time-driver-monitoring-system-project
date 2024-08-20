import cv2
import numpy as np
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import mediapipe as mp
from scipy.spatial import distance as dist
from threading import Thread
import pyttsx3
import playsound


mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 20
DISTRACTION_THRESH = 0.3  

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
FACE_LANDMARKS_IDX = [10, 234, 454, 10]  

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def sound_alarm():
    playsound.playsound('alert.wav')

def speak_message(message):
    engine = pyttsx3.init()
    engine.say(message)
    engine.runAndWait()

def face_aspect_ratio(face_landmarks):
    if len(face_landmarks) < 4:
        return 0
    A = dist.euclidean(face_landmarks[0], face_landmarks[1])
    B = dist.euclidean(face_landmarks[2], face_landmarks[3])
    return abs(A - B) / (A + B)

class DriverMonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Système de Surveillance du Conducteur")
        self.root.geometry("800x600")

        self.counter = 0
        self.alarm_on = False

        self.awake_icon = Label(root, text="Awake", fg="white", bg="blue", font=("Arial", 16), width=15)
        self.awake_icon.pack(pady=10)

        self.phone_icon = Label(root, text="Phone", fg="white", bg="blue", font=("Arial", 16), width=15)
        self.phone_icon.pack(pady=10)

        self.distracted_icon = Label(root, text="Focused", fg="white", bg="blue", font=("Arial", 16), width=15)
        self.distracted_icon.pack(pady=10)

        self.video_label = Label(root)
        self.video_label.pack()

        self.cap = cv2.VideoCapture(0)

        self.face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.update_video()

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_h, img_w = frame.shape[:2]

            results_face_mesh = self.face_mesh.process(rgb_frame)
            results_hands = self.hands.process(rgb_frame)

            if results_face_mesh.multi_face_landmarks:
                for face_landmarks in results_face_mesh.multi_face_landmarks:
                    landmarks = np.array([(lm.x, lm.y) for lm in face_landmarks.landmark])
                    
                    left_eye = landmarks[LEFT_EYE_IDX]
                    right_eye = landmarks[RIGHT_EYE_IDX]

                    left_ear = eye_aspect_ratio(left_eye)
                    right_ear = eye_aspect_ratio(right_eye)
                    ear = (left_ear + right_ear) / 2.0
                    
                    if ear < EYE_AR_THRESH:
                        self.counter += 1
                        if self.counter >= EYE_AR_CONSEC_FRAMES:
                            if not self.alarm_on:
                                self.alarm_on = True
                                Thread(target=sound_alarm, daemon=True).start()
                                Thread(target=speak_message, args=("You seem drowsy. Please take a break.",), daemon=True).start()
                            self.awake_icon.configure(bg="red", text="Drowsy")
                    else:
                        self.counter = 0
                        self.alarm_on = False
                        self.awake_icon.configure(bg="blue", text="Awake")

                    for (x, y) in left_eye:
                        cv2.circle(frame, (int(x * img_w), int(y * img_h)), 1, (0, 255, 0), -1)
                    for (x, y) in right_eye:
                        cv2.circle(frame, (int(x * img_w), int(y * img_h)), 1, (0, 255, 0), -1)

                    face_aspect = face_aspect_ratio(landmarks[FACE_LANDMARKS_IDX])
                    if face_aspect > DISTRACTION_THRESH:
                        self.distracted_icon.configure(bg="red", text="Distracted")
                        Thread(target=speak_message, args=("Please focus on the road.",), daemon=True).start()
                    else:
                        self.distracted_icon.configure(bg="blue", text="Focused")

            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    h, w, _ = frame.shape
                    for lm in hand_landmarks.landmark:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        
                        if results_face_mesh.multi_face_landmarks:
                            face_landmarks = np.array([(lm.x * w, lm.y * h) for lm in results_face_mesh.multi_face_landmarks[0].landmark])
                            if (cx > face_landmarks[:, 0].min() and cx < face_landmarks[:, 0].max() and
                                cy > face_landmarks[:, 1].min() and cy < face_landmarks[:, 1].max()):
                                self.phone_icon.configure(bg="red", text="Phone Detected")
                                Thread(target=speak_message, args=("Using phone. Please put it away.",), daemon=True).start()
                            else:
                                self.phone_icon.configure(bg="blue", text="No Phone")
                    
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_video)

    def __del__(self):
        self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = DriverMonitorApp(root)
    root.mainloop()
