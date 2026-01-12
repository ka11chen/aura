import os

from numpy.matlib import empty

from agent_loader import load_agent_from_json
from pipeline import run_pipeline
import json

judge_roster = [
    {"id": "Judge_Steve_Jobs", "target_figure": "Steve Jobs"},
    {"id": "Judge_Donald_Trump", "target_figure": "Donald Trump"},
    {"id": "Judge_Elon_Musk", "target_figure": "Elon Musk"},
]

async def main(landmark_ret, h, w):
    feature_extractor = load_agent_from_json("../agents/Feature_Extractor.json")
    score_aggregator = load_agent_from_json("../agents/Score_Aggregator.json")

    judges = []

    for judge in judge_roster:
        agent = load_agent_from_json("../agents/Judge.json")

        agent.label = judge["target_figure"]
        agent.name = judge["id"]

        judges.append(agent)

    if landmark_ret:
        save_landmarks_to_file(landmark_ret[3])

    result = await run_pipeline(
        feature_extractor=feature_extractor,
        judges=judges,
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