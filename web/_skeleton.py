import cv2

def draw_skeleton(img, landmarks, color_theme="default"):
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

    ideal_landmarks = []
    for i in range(33):
        if i in ideal_pts:
            ideal_landmarks.append({'x': ideal_pts[i][0], 'y': ideal_pts[i][1], 'visibility': 1.0})
        else:
            ideal_landmarks.append({'x': 0, 'y': 0, 'visibility': 0.0})
    return ideal_landmarks