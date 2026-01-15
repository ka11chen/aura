from agent_loader import load_agent_from_json
from pipeline import run_pipeline
from landmarks_to_json import save_landmarks_to_file

judge_roster = [
    {"id": "Judge_Steve_Jobs", "target_figure": "Steve Jobs"},
    {"id": "Judge_Donald_Trump", "target_figure": "Donald Trump"},
    # {"id": "Judge_Elon_Musk", "target_figure": "Elon Musk"},
]

async def main(landmark_ret, h, w):
    feature_extractor = load_agent_from_json("../agents/Feature_Extractor.json")
    score_aggregator = load_agent_from_json("../agents/Score_Aggregator.json")

    judges = []

    for judge in judge_roster:
        agent = load_agent_from_json("../agents/Judge.json")

        agent.label = judge["target_figure"]
        agent._name = judge["id"]

        judges.append(agent)

    if landmark_ret:
        save_landmarks_to_file(landmark_ret[3])

    result = await run_pipeline(
        feature_extractor=feature_extractor,
        judges=judges,
        aggregator=score_aggregator,
    )
    print("=====ULTIMATE RESULT=====")
    print(repr(result))
    return result