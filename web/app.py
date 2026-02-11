from flask import Flask, render_template, Response, jsonify, request, redirect
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
from landmarks_to_json import save_landmarks_to_file

from edit_pose import run_pose_edit
from evaluate import evaluate_pose

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
modified_skel = {}
judges=["Steve Jobs","Donald Trump"]

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
with open(PREFERENCE_FILE,'w') as f: # init preference
    preference={i:1 for i in judges}
    json.dump(preference,f)



def gen_landmark(frame, idx): 
    global done_cnt, landmark_dict
    try:
        filename = f"{SAVE_DIR}/frame_{idx}.jpg"
        cv2.imwrite(filename, frame)
        
        ret = mp_landmark.get_landmark(frame)
        landmark_dict[idx]=ret
        save_landmarks_to_file(ret,f"landmark_{idx+1}.json")
    except Exception as e:
        print(f"gen_landmark Error: {e}")
    finally:
        done_cnt += 1

def gen_modified_skels():
    global done_cnt, modified_skel, suggestion, state

    # print(landmark_dict.get(0))
    try:
        good_idx = []

        for idx in range(done_cnt):
            print("modified skel:", idx)
            filename = f"{SAVE_DIR}/frame_{idx}.jpg"
            if not os.path.exists(filename):
                print(f"gen_modified_skel: cannot find file {filename}")
                return
            if idx not in modified_skel:
                new_skel = run_pose_edit(filename, suggestion[0]["suggestion"])
                score = evaluate_pose(landmark_dict.get(idx), new_skel)
                print("idx: "+str(idx))
                print("score: "+str(score))
                if score >= 0.8:
                    modified_skel[idx] = new_skel
                    good_idx.append(idx)

        # set absence modified_skel to the closest one
        for i in range(len(good_idx)):
            idx = good_idx[i]
            if i == 0:
                for j in range(idx):
                    modified_skel[j] = modified_skel[idx]
                    #print(j, idx)
            else:
                lst_idx = good_idx[i-1]
                mid = (lst_idx + idx + 1) // 2
                for j in range(lst_idx + 1, mid):
                    modified_skel[j] = modified_skel[lst_idx]
                    #print(j, lst_idx)
                for j in range(mid, idx):
                    modified_skel[j] = modified_skel[idx]
                    #print(j, idx)
        for i in range(good_idx[-1]+1, done_cnt):
            modified_skel[i] = modified_skel[good_idx[-1]]
            #print(i, good_idx[-1])

        # candidates.sort(key=lambda x: x["score"], reverse=True)
        #
        # tops = candidates[:3] # get top 3, can change later
        #
        # modified_skel.clear()
        # for item in tops:
        #     modified_skel[item["idx"]] = item["skeleton"]
    finally:
        state = 4
    return

def normalize_preference():
    global judges

    with open(PREFERENCE_FILE, 'r') as f:
        prefs = json.load(f)

    prefs = {k: v for k, v in prefs.items() if k in judges}

    mx = max(prefs.values(), default=0)
    if mx == 0: mx = 1

    for k in prefs:
        prefs[k] /= mx

    with open(PREFERENCE_FILE, 'w') as f:
        json.dump(prefs, f, indent=2)

def gen_suggestion():
    global state, suggestion, judges
    try:
        raw_result = json.loads(asyncio.run(main(judges)))
        # raw_result=[{"suggestion":"Narrow steeple fingertip gap","severity":3,"description":"Steve Jobs: Your fingertips are too wide—bring the index fingertips into a tight V and reduce fingertip distance toward ~0.12–0.34, especially at the beginning and end.","judge":"Steve Jobs"},{"suggestion":"Maintain consistent hand height","severity":1,"description":"Steve Jobs: Wrists start high then drop below chest—keep hands roughly 0.09–0.30 units above shoulder height throughout, particularly mid and late.","judge":"Steve Jobs"},{"suggestion":"Soften elbow angle to ~105°","severity":2,"description":"Steve Jobs: Elbows are over-extended (up to 132°); relax into a gentle ~105° bend so arms read open but not locked.","judge":"Steve Jobs"},{"suggestion":"Set hand-span to ~1.9× shoulder width","severity":3,"description":"Donald Trump: Your hand-span collapses then over-stretches—open to about 1.9× shoulder width at the start and hold that span consistently.","judge":"Donald Trump"},{"suggestion":"Hold steeple angle at 80–95°","severity":3,"description":"Donald Trump: Steeple angle is inconsistent (too sharp then too flat); form a controlled triangular steeple around 80–95° in the opening and maintain it.","judge":"Donald Trump"},{"suggestion":"Stand more upright; limit forward lean","severity":3,"description":"Donald Trump: You lean forward too much (torso angle drops below ~160°); adopt a near-vertical posture (~172°) and check mid-speech and near the close to avoid pitching forward.","judge":"Donald Trump"}]
        normalize_preference()
        with open(PREFERENCE_FILE, 'r') as f:
            prefs = json.load(f)

        for data in raw_result:
            judge = data["judge"]
            if judge in prefs:
                data["severity"] = round(data["severity"] * prefs[judge], 2)
            else:
                prefs[judge] = 1
        
        with open(PREFERENCE_FILE, 'w') as f:
            json.dump(prefs, f, indent=2)
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
    gen_modified_skels()

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
    global state, start_time, last_saved_time, image_cnt, landmark_dict, done_cnt, suggestion, modified_skel
    landmark_dict.clear()
    state = 1
    suggestion = []
    modified_skel.clear()
    start_time = time.time()
    last_saved_time = start_time - SAVE_INTERVAL
    image_cnt = 0
    done_cnt = 0
    return jsonify({"status": "started"})

@app.route("/result_image/<img_type>/<int:frame_idx>")
def get_result_image(img_type, frame_idx):
    global landmark_dict, modified_skel
    
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
    has_data = res
    current_landmarks = res.pose_landmarks[0] if has_data else []

    if img_type == "skeleton":
        if not has_data:
            cv2.putText(black_canvas, "No Data", (50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (100,100,100), 2)
        else:
            draw_skeleton(original_img, current_landmarks, "default")
            black_canvas=original_img
    elif img_type == "modified":
        if not (frame_idx in modified_skel):
            cv2.putText(black_canvas, "Waiting or No Data", (50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (100,100,100), 2)
        else:
            ideal_landmarks = modified_skel[frame_idx].pose_landmarks[0]
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
    global state, image_cnt, done_cnt, landmark_dict, suggestion, modified_skel
    image_cnt = 0
    done_cnt = 0
    landmark_dict.clear()
    suggestion = []
    modified_skel.clear()
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

@app.route("/reference")
def reference():
    return render_template("reference.html")

@app.route("/load_judge", methods=["POST"])
def load_judge():
    global judges
    print("load: ",judges)
    return {"judges":judges}

@app.route("/delete_judge", methods=["POST"])
def delete_judge():
    global judges
    data = request.get_json()
    judge = data.get("judge")
    print("Deleting:", judge)
    judges.remove(judge)
    return jsonify(success=True)

@app.route("/add_judge", methods=["POST"])
def add_judge():
    global judges
    name = request.form.get("name")
    files = request.files.getlist("refimg[]")
    judges.append(name)
    print("name:",name)
    for idx,img in enumerate(files, start=1):
        file_bytes = img.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        filename=f"Judge_{name.replace(' ','_')}_{idx}.json"
        tmp=mp_landmark.get_landmark(frame)
        print(tmp)
        save_landmarks_to_file(tmp,filename,True)

    return redirect("/reference")

if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)
    mp_landmark.shutdown()
    cap.shutdown()
