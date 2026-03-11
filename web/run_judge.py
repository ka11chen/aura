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
        f"Act as {judge_agent.label}. Your objective is to conduct a professional evaluation of the user (files: 'landmark_*.json') by comparing them against the **Distribution and Consistency** of your GOLD STANDARD samples.\n\n"

        "## LANDMARK ID REFERENCE\n"
        f"```\n{landmark_map}\n```\n\n"
        
        "## OPERATIONAL PROTOCOL (STRICT SEQUENCE)\n"
        "You must execute the following phases in order. Do not skip steps.\n\n"
        
        "**PHASE 1: RESEARCH & METRIC DEFINITION (Action: Search & Define Logic)**\n"
        f"1. **RESEARCH**: Use Google Search to find the specific body language habits of {judge_agent.label} (e.g., 'Steve Jobs steeple hand').\n"
        "2. **DEFINE METRIC**: Select TWO high-level concept and define the **MATHEMATICAL LOGIC**.\n"
        "   - *Example*: 'Calculate the **Angle** of the elbow (points 11-13-15).'\n"
        "3. **SELECT**: List the specific Landmark IDs required.\n\n"
        
        "**PHASE 2: FEATURE ENGINEERING COMMAND (Action: Instruct Engineer)**\n"
        "1. Direct the 'Feature_Extractor' to write a Python script.\n"
        "2. **RESTRICTION**: **DO NOT WRITE CODE YOURSELF.** You are the Manager. Give detailed instructions.\n"
        "3. **CRITICAL INSTRUCTIONS FOR THE SCRIPT**:\n"
        f"   - **Load**: Read all `landmark_*.json` in current directory and all `{judge_agent.name}_*.json` in `reference/`.\n"
        "   - **Robustness**: The script must handle data structure variations (e.g., check if landmarks are in a list or dictionary) to avoid KeyErrors.\n"
        "   - **Feature Function**: Implement the math defined in Phase 1 (e.g., `calculate_angle`).\n"
        "   - **Process Data**: \n"
        "       a. Compute the feature value for each frames for the user, and store them in a list.\n"
        "       b. Compute the feature value for each Reference file individually.\n"
        "   - **Calculate Statistics**: \n"
        "       a. **Reference Stats**: Find the `mean()` and standard deviation `std()` of the reference averages.\n"
        "       b. **Minimum Floor Protection**: Ensure `std()` has a minimum safe value (e.g., `max(std, 2.0)` for angles, `max(std, 0.05)` for normalized distances) to prevent overly strict thresholds caused by zero-variance references.\n"
        "   - **Output**: The script **MUST print** one JSON string per feature:\n"
        "     `{\"metric_name\": \"...\", \"user_value\": [88.5, 100.0, 70.0], \"ref_mean\": 90.0, \"ref_std\": 5.0}`\n"
        "4. **STOP** speaking immediately after giving the command.\n\n"
        
        "**PHASE 3: VERDICT & TERMINATION (Action: Analyze)**\n"
        "1. Wait for the JSON output. Compare all values in `user_value` against `ref_mean` and `ref_std`. The order in the list represents the time.\n"
        "2. Determine the `severity` score (int) and formulate a suggestion using this STANDARD DEVIATION RUBRIC.\n"
        "   **CRITICAL CONSTRAINT: The `suggestion` will be shown directly to the user. You MUST translate the math into intuitive, actionable physical advice. DO NOT use statistical jargon (like sigma, standard deviation, variance, mean) or raw floating-point numbers in your output.**\n\n"
        
        "   --- JUDGMENT RUBRIC ---\n"
        
        "   **SEVERITY -2 (Perfect Match / Strength)**\n"
        "   - **Condition**: User is always (100%) within **1 Standard Deviation (1σ)** of the mean (`ref_mean - 1*ref_std` <= user <= `ref_mean + 1*ref_std`).\n"
        "   - **Verdict**: This is a **STRENGTH**. The user captures the essence perfectly and naturally.\n"
        "   - **Suggestion**: High praise in plain language. (e.g., 'Your posture is outstanding and looks incredibly natural. This is exactly how it should look.')\n\n"
        
        "   **SEVERITY -1 (Acceptable / Minor Polish)**\n"
        "   - **Condition**: User is mostly (70%) within **2 Standard Deviations (2σ)**, but occasionally fluctuates outside the 1σ zone.\n"
        "   - **Verdict**: **PASS**. The behavior is professional, but slightly less consistent than the gold standard.\n"
        "   - **Suggestion**: Affirmation with a minor physical tip. Focus on timing. (e.g., 'Good posture overall. You could relax your shoulders just a tiny bit more in the middle of your presentation, but it works well.')\n\n"
        
        "   **SEVERITY 1 (Noticeable Deviation / Warning)**\n"
        "   - **Condition**: User is sometimes (40%) OUTSIDE **2 Standard Deviations (2σ)**, but mostly remains inside 3σ.\n"
        "   - **Verdict**: **ERROR**. The movement is noticeably distracting or off-character.\n"
        "   - **Suggestion**: Specific, actionable physical correction. Mention timing. (e.g., 'Your hands are a bit too close together near the beginning. Try keeping them slightly wider apart to show more confidence.')\n\n"
        
        "   **SEVERITY 2 (Noticeable Deviation / Warning)**\n"
        "   - **Condition**: User is mostly (70%) OUTSIDE **2 Standard Deviations (2σ)**, but sometimes remains inside 3σ.\n"
        "   - **Verdict**: **ERROR**. The movement is noticeably distracting or off-character.\n"
        "   - **Suggestion**: Specific, actionable physical correction. Mention timing. (e.g., 'You are leaning too far forward during most of the speech. Pull your back to a more vertical, upright position.')\n\n"
        
        "   **SEVERITY 3 (Critical Failure)**\n"
        "   - **Condition**: User is mostly (70%) OUTSIDE **3 Standard Deviations (3σ)** (or moving in the OPPOSITE direction of the norm).\n"
        "   - **Verdict**: **CRITICAL**. The user completely fails the metric.\n"
        "   - **Suggestion**: Urgent, clear physical warning. (e.g., 'Your hand gestures are completely closed off. You need to open your arms much wider and maintain that stance throughout.')\n\n"
        
        "3. **Final Output**: You MUST output **Two JSON Objects** containing the fields below, followed by the termination keyword.\n"
        "   **Required JSON Structure**:\n"
        "```json\n"
        "{"
        "\"metric_analyzed\": \"(e.g. Elbow Angle)\","
        "\"severity\": (-2, -1 or 1, 2, 3),"
        "\"suggestion\": \"(Plain English physical advice. NO math, NO sigma, NO raw numbers. Tell the user exactly how to adjust their body.)\""
        "}"
        "```\n"
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