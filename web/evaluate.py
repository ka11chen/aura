import numpy as np
import math

def get_point(landmarks, idx):
    try:
        lm = landmarks.pose_landmarks[0].landmark[idx]
        return np.array([lm.x, lm.y])
    except:
        return None

torso_indices = [11, 12, 23, 24]

def evaluate_pose(original_landmarks, modified_landmarks):
    score = 1.0
    valid_points = 0

    for idx in torso_indices:
        p_orig = get_point(original_landmarks, idx)
        p_mod = get_point(modified_landmarks, idx)

        if p_orig is not None and p_mod is not None:
            dist = np.linalg.norm(p_orig - p_mod)
            score -= dist
            valid_points += 1

    score -= 4 - valid_points

    return score