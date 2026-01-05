import asyncio
import os

from numpy.matlib import empty

from agent_loader import load_agent_from_json
from pipeline import run_pipeline
import json

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
    work_dir = ".coding"

    file_path = os.path.join(work_dir, filename)

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

    with open(file_path, "w") as f:
        json.dump(simplified_data, f)