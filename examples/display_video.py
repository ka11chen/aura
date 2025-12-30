import cv2
import mediapipe as mp
import time

# ===== MediaPipe 設定 =====
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = 'model/pose_landmarker_full.task'

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE
)


# ===== 開啟影片 =====
VIDEO_PATH = 'assets/mv.mp4'   # ← 改成你的影片路徑
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
# print(fps)
prvtime=time.time_ns()

if not cap.isOpened():
    print("Cannot open video")
    exit()

with PoseLandmarker.create_from_options(options) as landmarker:
    while True:
        print("fps:",10**9/(time.time_ns()-prvtime))
        prvtime=time.time_ns()
        ret, frame = cap.read()
        if not ret:
            break   # 影片結束
        # frame=cv2.resize(
        #     frame,
        #     None,
        #     fx=0.5,
        #     fy=0.5,
        #     interpolation=cv2.INTER_LINEAR
        # )
        h, w = frame.shape[:2]
        # OpenCV(BGR) → MediaPipe(RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        # ===== Pose 偵測 =====
        result = landmarker.detect(mp_image)

        # ===== 畫 landmark =====
        if result.pose_landmarks:
            for lm in result.pose_landmarks[0]:
                cx = int(lm.x * w)
                cy = int(lm.y * h)
                if lm.visibility>0.5: cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                else: cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        # ===== 播放 =====
        cv2.imshow("Pose Video", frame)

        if cv2.waitKey(10) == ord('q'):
            break
        

cap.release()
cv2.destroyAllWindows()
