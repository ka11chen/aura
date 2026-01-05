from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.messages import TextMessage

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

async def run_analysis_session(feature_extractor_agent, judge_agent):
    termination = TextMentionTermination("TERMINATE_SESSION")

    team = RoundRobinGroupChat(
        participants=[judge_agent, feature_extractor_agent],
        termination_condition=termination,
        max_turns=6
    )

    task = (
        f"You are {judge_agent.name}. Please evaluate the speaker based on the data in 'landmarks.json'.\n"
        "Use these ID numbers to instruct the Feature Extractor on exactly what to measure:\n"
        f"```\n{landmark_map}\n```\n\n"
        "You can instruct the Feature Extractor to write Python scripts to calculate any specific features or metrics you prioritize.\n"
        "When you are done, output the word that consists of 'TERMINATE' and 'SESSION' joined by an underscore."
    )

    print(f"--- Running Session: {judge_agent.name} ---")

    result = await team.run(task=task)

    for msg in reversed(result.messages):
        if isinstance(msg, TextMessage) and msg.source == judge_agent.name:
            final_comment = msg.content
            break

    return final_comment.replace("TERMINATE_SESSION", "").strip()