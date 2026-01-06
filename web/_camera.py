import cv2
class VideoCamera(object):
    def __init__(self):
        #由opencv來獲取預設為0 裝置影像
        print("start camera")
        self.video = cv2.VideoCapture(0)

    def shutdown(self):
        print("del camera")
        self.video.release()

    def get_cam(self):
        ret, frame = self.video.read()
        return ret,frame