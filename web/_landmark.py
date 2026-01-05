import mediapipe as mp
import cv2

class landmark(object):
    def __init__(self):
        print("start mediapipe")
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        model_path='../model/pose_landmarker_full.task'

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE)
    
        self.landmarker=PoseLandmarker.create_from_options(options)

    def __del__(self):
        print("del mediapipe")
        self.landmarker.close()

    def get_landmark(self,frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        ret = self.landmarker.detect(mp_image)
        # print(ret)
        return ret
