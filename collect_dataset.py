import cv2
import mediapipe as mp
import csv
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

gestures = ['left', 'right', 'up', 'down', 'open_palm', 'neutral']
current_gesture = 0
samples_per_gesture = 85

with open('data/gesture_dataset.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    header = [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)] + [f'z{i}' for i in range(21)] + ['label']
    writer.writerow(header)

cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

print(f"Начинаем запись жеста: {gestures[current_gesture]}")
print("Нажми 'SPACE' для записи кадра, 'N' для следующего жеста, 'Q' для выхода")

sample_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    cv2.putText(frame, f"Gesture: {gestures[current_gesture]} ({sample_count}/{samples_per_gesture})", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    cv2.imshow('Data Collection', frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord(' ') and results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark
        
        wrist = landmarks[0]
        row = []
        for lm in landmarks:
            row.extend([lm.x, lm.y, lm.z])
        row.append(gestures[current_gesture])
        
        with open('data/gesture_dataset.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        sample_count += 1
        print(f"Записано: {sample_count}/{samples_per_gesture}")
        
        if sample_count >= samples_per_gesture:
            current_gesture += 1
            sample_count = 0
            if current_gesture >= len(gestures):
                print("Все жесты записаны!")
                break
            print(f"\nПереходим к жесту: {gestures[current_gesture]}")
    
    elif key == ord('n'):
        current_gesture = (current_gesture + 1) % len(gestures)
        sample_count = 0
        print(f"\nПереходим к жесту: {gestures[current_gesture]}")
    
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()