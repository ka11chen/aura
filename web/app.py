from flask import Flask, render_template, Response, jsonify, request
import cv2
import time
import os
import threading
import asyncio
import numpy as np

import json
from _autogen import main
from _camera import VideoCamera
from _landmark import landmark
from _skeleton import *

app = Flask(__name__)

cap = VideoCamera()
mp_landmark = landmark()

CAPTURE_DURATION = 10
SAVE_INTERVAL = 1
SAVE_DIR = "captures"
UPLOAD_DIR = "uploads"
PREFERENCE_FILE = "user_preferences.json"

state = 0
start_time = None
last_saved_time = None
image_cnt = 0
done_cnt = 0
landmark_dict = {}
suggestion = []
judges=["Steve Jobs","Donald Trump"] # should enable user judge later

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
with open(PREFERENCE_FILE,'w') as f: # init preference
    preference={i:1 for i in judges}
    json.dump(preference,f)

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

def gen_suggestion():
    global state, suggestion, landmark_dict
    landmark_list = [landmark_dict[i] for i in sorted(landmark_dict.keys())]
    try:
        raw_result = json.loads(asyncio.run(main(landmark_list)))
        # raw_result=[]
        with open(PREFERENCE_FILE, 'r') as f:
            prefs = json.load(f)
            for data in raw_result:
                data["severity"]*=prefs[data["judge"]]
        suggestion = sorted(raw_result, key=lambda x: x["severity"],reverse=True)

    except Exception as e:
        print(f"gen_suggestion Error: {e}")
        suggestion = [{
            "judge": "System",
            "suggestion": "System Error",
            "severity": 1.0,
            "description": "後端分析發生錯誤，請檢查後端日誌。"
        }]
        
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

            if now - start_time >= CAPTURE_DURATION:
                state=10 # so that /upload can use
        
        if state == 10:
            if done_cnt >= image_cnt:
                state = 2
                threading.Thread(target=gen_suggestion, daemon=True).start()

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

@app.route("/update_preferences", methods=["POST"])
def update_preferences():
    from flask import request
    try:
        data = request.json
        if not isinstance(data, list):
            return jsonify({"status": "error", "message": "Invalid format"}), 400
            
        with open(PREFERENCE_FILE, 'w') as f:
            json.dump(data, f)
            
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/start_capture", methods=["POST"])
def start_capture():
    global state, start_time, last_saved_time, image_cnt, landmark_dict, done_cnt, suggestion
    landmark_dict.clear()
    state = 1
    suggestion = []
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
            draw_skeleton(original_img, current_landmarks, "default")
            black_canvas=original_img
        elif img_type == "modified":
            ideal_landmarks = generate_ideal_pose(current_landmarks)
            draw_skeleton(black_canvas, ideal_landmarks, "ideal")

    _, img_encoded = cv2.imencode('.jpg', black_canvas)
    return Response(img_encoded.tobytes(), mimetype='image/jpeg')

@app.route("/upd_preference", methods=["POST"])
def upd_preference():
    try:
        data = request.get_json()
        judge = data.get("judge")
        delta = data.get("delta")

        with open(PREFERENCE_FILE, 'r') as f:
            prefs = json.load(f)

        if judge in prefs:
            prefs[judge] = max(0, prefs[judge] + delta * 0.2) # discuss this later

        with open(PREFERENCE_FILE, 'w') as f:
            json.dump(prefs, f)

        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/upload", methods=["POST"])
def upload():
    global state, image_cnt, done_cnt
    image_cnt = 0
    done_cnt = 0
    file = request.files.get("file")
    if not file:
        return "No file", 400
    
    # Save uploaded video
    video_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(video_path)

    cap2 = cv2.VideoCapture(video_path)
    if not cap2.isOpened():
        return "Failed to open video", 400

    fps = cap2.get(cv2.CAP_PROP_FPS)

    frame_interval = int(fps * SAVE_INTERVAL)
    max_frames = int(fps * CAPTURE_DURATION)

    frame_idx = 0
    current_idx = 0

    while cap2.isOpened() and frame_idx < max_frames:
        ret, frame = cap2.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            current_idx = image_cnt
            image_cnt+=1
            threading.Thread(target=gen_landmark, args=(frame.copy(), current_idx), daemon=True).start()

        frame_idx += 1

    cap2.release()
    state = 10 # call gen_suggestion in big loop
    return {"status": "started"}

if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)
    mp_landmark.shutdown()
    cap.shutdown()
