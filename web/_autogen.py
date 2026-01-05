import asyncio

from numpy.matlib import empty

from agent_loader import load_agent_from_json
from pipeline import run_pipeline
import json

landmark_map = """0 - nose
    1 - left eye (inner)
    2 - left eye
    3 - left eye (outer)
    4 - right eye (inner)
    5 - right eye
    6 - right eye (outer)
    7 - left ear
    8 - right ear
    9 - mouth (left)
    10 - mouth (right)
    11 - left shoulder
    12 - right shoulder
    13 - left elbow
    14 - right elbow
    15 - left wrist
    16 - right wrist
    17 - left pinky
    18 - right pinky
    19 - left index
    20 - right index
    21 - left thumb
    22 - right thumb
    23 - left hip
    24 - right hip
    25 - left knee
    26 - right knee
    27 - left ankle
    28 - right ankle
    29 - left heel
    30 - right heel
    31 - left foot index
    32 - right foot index"""

async def main(landmark_ret, h, w):
    feature_extractor = load_agent_from_json("../agents/Feature_Extractor.json")
    judge_steve = load_agent_from_json("../agents/Judge_Steve_Jobs.json")
    judge_trump = load_agent_from_json("../agents/Judge_Donald_Trump.json")
    score_aggregator = load_agent_from_json("../agents/Score_Aggregator.json")

    if landmark_ret:
        save_landmarks_to_file(landmark_ret[3])

    result = await run_pipeline(
        feature_extractor=feature_extractor,
        judges=[judge_steve, judge_trump],
        aggregator=score_aggregator,
    )

    return result


def save_landmarks_to_file(result, filename="landmarks.json"):
    simplified_data = []

    for frame in result.pose_landmarks:
        frame_points = []
        for p in frame:
            frame_points.append({
                "x": p.x,
                "y": p.y,
                "z": p.z,
                "visibility": getattr(p, 'visibility', 0.0)
            })
        simplified_data.append(frame_points)

    with open(filename, "w") as f:
        json.dump(simplified_data, f)