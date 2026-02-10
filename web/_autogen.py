from agent_loader import load_agent_from_json
from pipeline import run_pipeline
# from landmarks_to_json import save_landmarks_to_file

# judge_roster = [
#     {"id": "Judge_Steve_Jobs", "target_figure": "Steve Jobs"},
#     {"id": "Judge_Donald_Trump", "target_figure": "Donald Trump"},
#     # {"id": "Judge_Elon_Musk", "target_figure": "Elon Musk"},
# ]

async def main(judge_name):
    # # test
    # test = '[{"suggestion":"Narrow steeple fingertip gap","severity":3,"description":"Steve Jobs: Your fingertips are too wide—bring the index fingertips into a tight V and reduce fingertip distance toward ~0.12–0.34, especially at the beginning and end.","judge":"Steve Jobs"},{"suggestion":"Maintain consistent hand height","severity":1,"description":"Steve Jobs: Wrists start high then drop below chest—keep hands roughly 0.09–0.30 units above shoulder height throughout, particularly mid and late.","judge":"Steve Jobs"},{"suggestion":"Soften elbow angle to ~105°","severity":2,"description":"Steve Jobs: Elbows are over-extended (up to 132°); relax into a gentle ~105° bend so arms read open but not locked.","judge":"Steve Jobs"},{"suggestion":"Set hand-span to ~1.9× shoulder width","severity":3,"description":"Donald Trump: Your hand-span collapses then over-stretches—open to about 1.9× shoulder width at the start and hold that span consistently.","judge":"Donald Trump"},{"suggestion":"Hold steeple angle at 80–95°","severity":3,"description":"Donald Trump: Steeple angle is inconsistent (too sharp then too flat); form a controlled triangular steeple around 80–95° in the opening and maintain it.","judge":"Donald Trump"},{"suggestion":"Stand more upright; limit forward lean","severity":3,"description":"Donald Trump: You lean forward too much (torso angle drops below ~160°); adopt a near-vertical posture (~172°) and check mid-speech and near the close to avoid pitching forward.","judge":"Donald Trump"}]'
    # return test

    feature_extractor = load_agent_from_json("../agents/Feature_Extractor.json")
    score_aggregator = load_agent_from_json("../agents/Score_Aggregator.json")

    judges = []

    for judge in judge_name:
        agent = load_agent_from_json("../agents/Judge.json")

        agent.label = judge
        agent._name = f"Judge_{judge.replace(' ','_')}"

        judges.append(agent)

    result = await run_pipeline(
        feature_extractor=feature_extractor,
        judges=judges,
        aggregator=score_aggregator,
    )
    print("=====ULTIMATE RESULT=====")
    print(repr(result))
    return result