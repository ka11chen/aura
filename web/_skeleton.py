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