import math
import random


def get_pts(landmarks):
    pts = {}
    for i, lm in enumerate(landmarks):
        if isinstance(lm, dict):
            x, y = lm['x'], lm['y']
            vis = lm.get('visibility', 1.0)
        else:
            x, y = lm.x, lm.y
            vis = getattr(lm, 'visibility', 1.0)

        if vis > 0.3:
            pts[i] = (x, y)
    return pts

def calc_dist(lm1, lm2):
    return math.dist(lm1, lm2)

torso_indices = [11, 12, 23, 24]

def evaluate_pose(original_landmarks, modified_landmarks):
    # # test
    # return random.uniform(0.5, 1.0)

    score = 1.0
    valid_points = 0

    try:
        pts_orig = get_pts(original_landmarks.pose_landmarks[0])
    except:
        return -100

    try:
        pts_mod = get_pts(modified_landmarks.pose_landmarks[0])
    except:
        return -100

    for idx in torso_indices:
        p_orig = pts_orig[idx]
        p_mod = pts_mod[idx]

        if p_orig is not None and p_mod is not None:
            dist = calc_dist(p_orig, p_mod)
            score -= dist
            valid_points += 1

    score -= (4 - valid_points) * 100

    return score