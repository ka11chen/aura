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


def draw_polished_skeleton(img, landmarks, color_theme="default"):
    h, w, _ = img.shape
    pts = {}
    
    for i, lm in enumerate(landmarks):
        if isinstance(lm, dict):
            x, y = lm['x'], lm['y']
            vis = lm.get('visibility', 1.0)
        else:
            x, y = lm.x, lm.y
            vis = getattr(lm, 'visibility', 1.0)
        
        if vis > 0.3: 
            pts[i] = (int(x * w), int(y * h))

    if color_theme == "ideal":
        c_line = (100, 255, 100) 
        c_joint = (200, 255, 200)
    else:
        c_line = (255, 191, 0) 
        c_joint = (255, 255, 255)

    def draw_line(p1, p2, color):
        if p1 in pts and p2 in pts:
            cv2.line(img, pts[p1], pts[p2], color, 3, cv2.LINE_AA)

    def draw_joint(idx, color):
        if idx in pts:
            cv2.circle(img, pts[idx], 5, color, -1, cv2.LINE_AA)


    connections = [
        (11, 12), (23, 24), (11, 23), (12, 24), 
        (11, 13), (13, 15), (12, 14), (14, 16), 
        (23, 25), (25, 27), (24, 26), (26, 28)  
    ]
    
    for p1, p2 in connections:
        draw_line(p1, p2, c_line)


    for idx in pts:
        draw_joint(idx, c_joint)
    

    if 0 in pts:
        cv2.circle(img, pts[0], 15, c_line, 2, cv2.LINE_AA)

def generate_ideal_pose(ref_landmarks):

    if not ref_landmarks: return []
    xs = [lm.x for lm in ref_landmarks]
    ys = [lm.y for lm in ref_landmarks]
    if not xs or not ys: return []

    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
    center_x = (min_x + max_x) / 2
    height = max_y - min_y
    

    w_shoulder = height * 0.22
    w_hip = height * 0.15
    y_nose = min_y + height * 0.08
    y_shoulder = y_nose + height * 0.15
    y_hip = y_shoulder + height * 0.38
    

    ideal_pts = {
        0: (center_x, y_nose),
        11: (center_x + w_shoulder/2, y_shoulder), 12: (center_x - w_shoulder/2, y_shoulder),
        23: (center_x + w_hip/2, y_hip), 24: (center_x - w_hip/2, y_hip),
        13: (center_x + w_shoulder*0.9, y_shoulder + height*0.18), # 手肘微開
        14: (center_x - w_shoulder*0.9, y_shoulder + height*0.18),
        15: (center_x + w_shoulder*1.1, y_hip - height*0.05),     # 手腕自信高度
        16: (center_x - w_shoulder*1.1, y_hip - height*0.05),
        25: (center_x + w_hip/2, y_hip + height*0.3), 26: (center_x - w_hip/2, y_hip + height*0.3),
        27: (center_x + w_hip/2, max_y), 28: (center_x - w_hip/2, max_y)
    }

    final_landmarks = []
    for i in range(33):
        if i in ideal_pts:
            final_landmarks.append({'x': ideal_pts[i][0], 'y': ideal_pts[i][1], 'visibility': 1.0})
        else:
            final_landmarks.append({'x': 0, 'y': 0, 'visibility': 0.0})
    return final_landmarks


def gen(frame, idx): 
    global landmark_dict, done_cnt
    try:

        filename = f"{SAVE_DIR}/frame_{idx}.jpg"
        cv2.imwrite(filename, frame)
        

        ret = mp_landmark.gen_landmark(frame)
        landmark_dict[idx] = ret
    except Exception as e:
        print(f"Gen Error: {e}")
    finally:
        done_cnt += 1

def gen_suggestion(h, w):
    global state, suggestion, landmark_dict
    

    landmark_list = [landmark_dict[i] for i in sorted(landmark_dict.keys())]
    
    try:

        suggestion = str(asyncio.run(main(landmark_list, h, w)))
    except Exception as e:
        print(f"Autogen Error: {e}")
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
                threading.Thread(target=gen, args=(frame.copy(), current_idx), daemon=True).start()

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
            draw_polished_skeleton(black_canvas, current_landmarks, "default")
            cv2.putText(black_canvas, f"Frame {frame_idx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 1)
        elif img_type == "modified":
            ideal = generate_ideal_pose(current_landmarks)
            draw_polished_skeleton(black_canvas, ideal, "ideal")
            cv2.putText(black_canvas, "AI Ideal Pose", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,255,100), 1)

    _, img_encoded = cv2.imencode('.jpg', black_canvas)
    return Response(img_encoded.tobytes(), mimetype='image/jpeg')

if __name__ == "__main__":
    app.run(debug=True)
