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
        max_turns=12
    )

    key_part_1 = "TERMINATE"
    key_part_2 = "SESSION"

    task = (
        f"Act as {judge_agent.name}. Your objective is to conduct a professional evaluation of the user (file: 'landmarks.json') by comparing them against the **Range and Consistency** of your GOLD STANDARD samples.\n\n"

        "## LANDMARK ID REFERENCE\n"
        f"```\n{landmark_map}\n```\n\n"

        "## DATA LOCATIONS\n"
        "1. **User Data**: `landmarks.json` (Structure: A List of frames, where each frame has 'landmarks')\n"
        f"2. **Reference Data**: Folder `reference/` containing files like `{judge_agent.name}_1.json`, `{judge_agent.name}_2.json`.\n\n"

        "## OPERATIONAL PROTOCOL (STRICT SEQUENCE)\n"
        "You must execute the following phases in order. Do not skip steps.\n\n"

        "**PHASE 1: RESEARCH & METRIC DEFINITION (Action: Search & Define Logic)**\n"
        f"1. **RESEARCH**: Use Google Search to find the specific body language habits of {judge_agent.name} (e.g., 'Steve Jobs steeple hand', 'Trump accordion hands').\n"
        "2. **DEFINE METRIC**: Select ONE high-level concept and define the **MATHEMATICAL LOGIC**.\n"
        "   - *Example*: 'Calculate the **Angle** of the elbow (points 11-13-15).'\n"
        "3. **SELECT**: List the specific Landmark IDs required.\n\n"

        "**PHASE 2: FEATURE ENGINEERING COMMAND (Action: Instruct Engineer)**\n"
        "1. Direct the 'Feature_Extractor' to write a Python script.\n"
        "2. **RESTRICTION**: **DO NOT WRITE CODE YOURSELF.** You are the Manager. Only give instructions.\n"
        "3. **CRITICAL INSTRUCTIONS FOR THE SCRIPT**:\n"
        "   - **Load**: Read `landmarks.json` and ALL `{judge_agent.name}*.json` files in `reference/`.\n"
        "   - **Robustness**: The script must handle data structure variations (e.g., check if landmarks are in a list or dictionary) to avoid KeyErrors.\n"
        "   - **Feature Function**: Implement the math defined in Phase 1 (e.g., `calculate_angle`).\n"
        "   - **Process Data**: \n"
        "       a. Compute the average feature value across ALL frames for the User.\n"
        "       b. Compute the average feature value for EACH Reference file individually.\n"
        "   - **Calculate Statistics**: \n"
        "       a. **Reference Range**: Find `min()` and `max()` of the reference averages.\n"
        "       b. **Reference Mean**: Find `mean()` of the reference averages.\n"
        "   - **Output**: The script **MUST print** a JSON string:\n"
        "     `{\"metric_name\": \"...\", \"user_value\": 88.5, \"ref_min\": 80.0, \"ref_max\": 100.0, \"ref_mean\": 90.0, \"status\": \"Inside/Outside\"}`\n"
        "4. **STOP** speaking immediately after giving the command.\n\n"

        "**PHASE 3: VERDICT & TERMINATION (Action: Analyze)**\n"
        "1. Wait for the JSON output. Compare `user_value` against `ref_min` and `ref_max`.\n"
        "2. Determine the `severity` score (0 to 3) using this RANGE-BASED RUBRIC:\n\n"
        
        "   --- JUDGMENT RUBRIC (0-3 Scale) ---\n"
        
        "   **SEVERITY 0 (Perfect Match / Strength)**\n"
        "   - **Condition**: User is **INSIDE** the range (`ref_min` <= user <= `ref_max`) AND positioned near the center (Optimal).\n"
        "   - **Verdict**: This is a **STRENGTH**. The user captures the essence perfectly.\n"
        "   - **Suggestion**: High praise. (e.g., 'Your precision here is outstanding. This is exactly how it should look.')\n\n"
        
        "   **SEVERITY 1 (Acceptable / Minor Polish)**\n"
        "   - **Condition**: User is **INSIDE** the range, but located near the edges (borderline).\n"
        "   - **Verdict**: **PASS**. The behavior is professional and natural, though there is slight room for refinement.\n"
        "   - **Suggestion**: Affirmation with a minor tip. (e.g., 'Good posture. You could relax your shoulders just a tiny bit more, but it works.')\n\n"
        
        "   **SEVERITY 2 (Noticeable Deviation / Warning)**\n"
        "   - **Condition**: User is **OUTSIDE** the range, but the distance to the edge is **LESS than 1.0x** the `range_span`.\n"
        "     *Formula*: `dist(user, edge) < range_span`\n"
        "   - **Verdict**: **ERROR**. The movement is distracting or slightly off-character.\n"
        "   - **Suggestion**: Specific correction. (e.g., 'You are leaning too far forward. Pull back to vertical.')\n\n"
        
        "   **SEVERITY 3 (Critical Failure)**\n"
        "   - **Condition**: User is **OUTSIDE** the range, and the distance is **MORE than 1.0x** the `range_span` (or moving in OPPOSITE direction).\n"
        "   - **Verdict**: **CRITICAL**. The user completely fails the metric.\n"
        "   - **Suggestion**: Urgent warning. (e.g., 'Stop! This is completely wrong. You must reset your stance immediately.')\n\n"

        "2. **Final Output**: You MUST output a **Single JSON Object** containing the fields below, followed by the termination keyword.\n"
        "   **Required JSON Structure**:\n"
        "   ```json\n"
        "   {\n"
        "     \"metric_analyzed\": \"(e.g. Elbow Angle)\",\n"
        "     \"severity\": 1,  // Integer: 0, 1, 2, or 3\n"
        "     \"suggestion\": \"(Write your advice here based on the data difference)\"\n"
        "   }\n"
        "   ```\n"
        f"3. ONLY THEN, output the exact keyword consisting of '{key_part_1}' and '{key_part_2}' joined by an underscore.\n\n"
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