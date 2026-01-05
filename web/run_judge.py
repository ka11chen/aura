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
    term_key = "TERMINATE_SESSION"
    termination = TextMentionTermination(term_key)

    team = RoundRobinGroupChat(
        participants=[judge_agent, feature_extractor_agent],
        termination_condition=termination,
        max_turns=10
    )

    key_part_1 = "TERMINATE"
    key_part_2 = "SESSION"

    task = (
        f"Act as {judge_agent.name}. Your objective is to conduct a professional evaluation of the speaker based on 'landmarks.json'.\n\n"

        "## LANDMARK ID REFERENCE\n"
        f"```\n{landmark_map}\n```\n\n"

        "## OPERATIONAL PROTOCOL (STRICT SEQUENCE)\n"
        "You must execute the following phases in order. Do not skip steps.\n\n"

        "**PHASE 1: CRITERIA DEFINITION (Action: Search/Recall)**\n"
        f"1. First, identify the key body language principles of {judge_agent.name} (e.g., 'Openness', 'Dominance', 'Minimalism').\n"
        "2. If you have search tools, search for specific posture metrics associated with this persona. If not, rely on your core knowledge.\n"
        "3. Decide which specific landmarks (IDs) are needed to measure these principles.\n\n"

        "**PHASE 2: DATA EXTRACTION COMMAND (Action: Instruct Engineer)**\n"
        "1. Direct the 'Feature_Extractor' to write a Python script.\n"
        "2. **CRITICAL INSTRUCTION TO ENGINEER**: You must tell the Engineer two things:\n"
        "   - **Structure**: The data in 'landmarks.json' is a time-series list (`List[Frame] -> List[Point] -> Dict`). Iteration over frames is required.\n"
        "   - **Output**: The script **MUST use `print()`** to output the final calculated numbers. If nothing is printed, you cannot see the result.\n"
        "3. **STOP** speaking immediately after giving the command. Wait for the code execution.\n\n"

        "**PHASE 3: VERDICT & TERMINATION (Action: Analyze)**\n"
        "1. Wait until the Feature_Extractor returns the numerical output (e.g., 'Average Angle: 45.2').\n"
        "2. Compare these numbers against your Phase 1 criteria.\n"
        "3. Provide your final professional critique.\n"
        f"4. ONLY THEN, output the exact keyword consisting of '{key_part_1}' and '{key_part_2}' joined by an underscore.\n\n"
    )

    print(f"--- Running Session: {judge_agent.name} ---")
    result = await team.run(task=task)

    print(result)

    final_comment = ""

    for msg in reversed(result.messages):
        if isinstance(msg, TextMessage) and msg.source == judge_agent.name:
            final_comment = msg.content
            break

    return final_comment