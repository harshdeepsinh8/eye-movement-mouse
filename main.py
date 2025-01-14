import time
import cv2
import mediapipe as mp
import pyautogui
cam=cv2.VideoCapture(0)
face_mesh=mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w,screen_h=pyautogui.size() 

last_blink_time = 0  # To store the time of the last blink

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)
    landmark_points = result.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape
    if landmark_points:
        landmarks = landmark_points[0].landmark
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 2, (0, 255, 0))
            if id == 1:
                screen_x = screen_w / frame_w * x
                screen_y = screen_h / frame_h * y
                pyautogui.moveTo(screen_x, screen_y)

        left = [landmarks[145], landmarks[159]]
        for landmark in left:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 2, (0, 255, 255))

        # Blink detection
        if (left[0].y - left[1].y) < 0.004:  # Blink detected
            current_time = time.time()
            if (current_time - last_blink_time) < 2:  # Double click threshold
                pyautogui.doubleClick()
                last_blink_time = 0  # Reset to avoid triple-click issues
            else:
                pyautogui.click()
                last_blink_time = current_time
            pyautogui.sleep(0.2)  # Small delay to avoid rapid repeated clicks

    cv2.imshow('Eye Mouse', frame)
    key = cv2.waitKey(1)
    if key == 27:  # Press ESC to exit
        break
