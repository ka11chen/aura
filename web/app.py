from flask import Flask, render_template, Response, jsonify
import cv2
import time
import os
import threading

from main import main
from _camera import VideoCamera
from _landmark import landmark

app = Flask(__name__)

cap = VideoCamera()
mp = landmark()
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

CAPTURE_DURATION = 10      # 秒
SAVE_INTERVAL = 1          # 每幾秒存一張

state = 0
start_time = None
last_saved_time = None
image_cnt = 0
done_cnt=0
landmark_ret=[]
suggestion=''

SAVE_DIR = "captures"
os.makedirs(SAVE_DIR, exist_ok=True)

def gen(frame): # background
    global image_cnt, landmark_ret, done_cnt

    print(f"get pic {image_cnt}")
    filename = f"{SAVE_DIR}/frame_{image_cnt}.jpg"
    cv2.imwrite(filename, frame)
    ret=mp.gen_landmark(frame)
    landmark_ret.append(ret)
    done_cnt +=1
    # return mp.gen_landmark(frame)

def gen_suggestion(landmark_ret,h,w):
    global state,suggestion
    suggestion=str(main(landmark_ret,h,w))
    state=3

def gen_frames():
    global state, start_time, last_saved_time, landmark_ret, done_cnt, image_cnt

    while True:
        success, frame = cap.get_cam()
        if not success:
            return

        now = time.time()

        if state==1:
            # 每 1 秒存一張
            if now - last_saved_time >= SAVE_INTERVAL and now - start_time <= CAPTURE_DURATION:
                last_saved_time = now
                image_cnt+=1
                threading.Thread(target=gen,args=(frame,),daemon=True).start()

            # 超過 10 秒停止
            if now - start_time >= CAPTURE_DURATION and done_cnt==image_cnt:
                state = 2
                h, w = frame.shape[:2]
                threading.Thread(target=gen_suggestion,args=(landmark_ret,h,w),daemon=True).start()
                gen_suggestion(landmark_ret,h,w)

        ret, jpeg = cv2.imencode('.jpg', frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/status")
def status():
    return {
        "state": state,
        "suggestion": suggestion
    }

@app.route("/start_capture", methods=["POST"])
def start_capture():
    global state, start_time, last_saved_time, image_cnt, landmark_ret,done_cnt

    landmark_ret.clear()
    state = 1
    start_time = time.time()
    last_saved_time = start_time
    image_cnt = 0
    done_cnt=0
    return jsonify({"status": "started"})

if __name__ == "__main__":
    app.run(debug=True)