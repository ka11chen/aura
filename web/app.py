from flask import Flask, render_template, Response, jsonify
import cv2
import time
import os
import threading
import asyncio
import numpy as np

from _autogen import main
from _camera import VideoCamera
from _landmark import landmark
from _skeleton import *

app = Flask(__name__)

cap = VideoCamera()
mp_landmark = landmark()

CAPTURE_DURATION = 10
SAVE_INTERVAL = 1

state = 0
start_time = None
last_saved_time = None
image_cnt = 0
done_cnt = 0

landmark_dict = {}
suggestion = ''

SAVE_DIR = "captures"
os.makedirs(SAVE_DIR, exist_ok=True)

def gen_landmark(frame, idx): 
    global landmark_dict, done_cnt
    try:
        filename = f"{SAVE_DIR}/frame_{idx}.jpg"
        cv2.imwrite(filename, frame)
        
        ret = mp_landmark.get_landmark(frame)
        landmark_dict[idx] = ret
    except Exception as e:
        print(f"gen_landmark Error: {e}")
    finally:
        done_cnt += 1

def gen_suggestion(h, w):
    global state, suggestion, landmark_dict
    
    landmark_list = [landmark_dict[i] for i in sorted(landmark_dict.keys())]
    
    try:
        suggestion = str(asyncio.run(main(landmark_list, h, w)))
    except Exception as e:
        print(f"gen_suggestion Error: {e}")
        suggestion = "後端分析發生錯誤，請檢查後端日誌。"
        
    state = 3

def gen_frames():
    global state, start_time, last_saved_time, image_cnt, done_cnt

    while True:
        success, frame = cap.get_cam()
        if not success: return

        now = time.time()

        if state == 1:
            if now - last_saved_time >= SAVE_INTERVAL and now - start_time <= CAPTURE_DURATION:
                last_saved_time = now
                current_idx = image_cnt
                image_cnt += 1
                threading.Thread(target=gen_landmark, args=(frame.copy(), current_idx), daemon=True).start()

            if now - start_time >= CAPTURE_DURATION and done_cnt >= image_cnt:
                state = 2
                h, w = frame.shape[:2]
                threading.Thread(target=gen_suggestion, args=(h, w), daemon=True).start()

        ret, jpeg = cv2.imencode('.jpg', frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/status")
def status():
    return {
        "state": state,
        "suggestion": suggestion,
        "total_frames": len(landmark_dict)
    }

@app.route("/start_capture", methods=["POST"])
def start_capture():
    global state, start_time, last_saved_time, image_cnt, landmark_dict, done_cnt, suggestion
    landmark_dict.clear()
    state = 1
    suggestion = ''
    start_time = time.time()
    last_saved_time = start_time - SAVE_INTERVAL
    image_cnt = 0
    done_cnt = 0
    return jsonify({"status": "started"})

@app.route("/result_image/<img_type>/<int:frame_idx>")
def get_result_image(img_type, frame_idx):
    global landmark_dict
    
    filename = f"{SAVE_DIR}/frame_{frame_idx}.jpg"
    if not os.path.exists(filename):

        blank = np.zeros((480, 640, 3), np.uint8)
        _, img_encoded = cv2.imencode('.jpg', blank)
        return Response(img_encoded.tobytes(), mimetype='image/jpeg')
        
    original_img = cv2.imread(filename)
    h, w, _ = original_img.shape

    if img_type == "original":
        _, img_encoded = cv2.imencode('.jpg', original_img)
        return Response(img_encoded.tobytes(), mimetype='image/jpeg')

    black_canvas = np.zeros((h, w, 3), dtype=np.uint8)
    
    res = landmark_dict.get(frame_idx)
    has_data = res and res.pose_landmarks
    current_landmarks = res.pose_landmarks[0] if has_data else []

    if not has_data:
        cv2.putText(black_canvas, "No Data", (50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (100,100,100), 2)
    else:
        if img_type == "skeleton":
            draw_skeleton(black_canvas, current_landmarks, "default")
            cv2.putText(black_canvas, f"Frame {frame_idx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 1)
        elif img_type == "modified":
            ideal = generate_ideal_pose(current_landmarks)
            draw_skeleton(black_canvas, ideal, "ideal")
            cv2.putText(black_canvas, "AI Ideal Pose", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,255,100), 1)

    _, img_encoded = cv2.imencode('.jpg', black_canvas)
    return Response(img_encoded.tobytes(), mimetype='image/jpeg')

if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)
    mp_landmark.shutdown()
    cap.shutdown()
