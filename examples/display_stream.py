import mediapipe as mp
import cv2
import time

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = 'model/pose_landmarker_full.task'

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)

prvtime=time.time_ns()
with PoseLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        print("fps:",10**9/(time.time_ns()-prvtime))
        prvtime=time.time_ns()
        if not ret:
            print("Camera read failed")
            exit()

        h, w = frame.shape[:2]

        # BGR → RGB 給 MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = landmarker.detect(mp_image)

        if result.pose_landmarks:
            for keyPoint in result.pose_landmarks[0]:
                cx = int(keyPoint.x * w)
                cy = int(keyPoint.y * h)
                if keyPoint.visibility>0.5: cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                else: cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        cv2.imshow('test', frame)
        if cv2.waitKey(10) == ord('q'):      # 每一毫秒更新一次，直到按下 q 結束 # t=10 functional
            break
    
    cv2.destroyAllWindows()
    cap.release()
