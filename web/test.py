import asyncio
import cv2
from _landmark import landmark
from _autogen import main

mp = landmark()

def test():
    landmark_ret = []

    frame = cv2.imread("test.jpg")

    for i in range(10):
        ret = mp.get_landmark(frame)
        landmark_ret.append(ret)

    asyncio.run(main(landmark_ret, 1920, 1080))

if __name__ == "__main__":
    test()