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
        f"Act as {judge_agent.label}. Your objective is to conduct a professional evaluation of the user (file: 'landmarks.json') by comparing them against the **Range and Consistency** of your GOLD STANDARD samples.\n\n"

        "## LANDMARK ID REFERENCE\n"
        f"```\n{landmark_map}\n```\n\n"

        "## DATA LOCATIONS\n"
        "1. **User Data**: `landmarks.json` (Structure: A List of frames (`[...]`), where each frame is a LIST of landmark objects (`[{'x':..., 'y':..., 'z':...}, ...]`). DIRECTLY ACCESSIBLE, NO 'landmarks' key.)\n"
        f"2. **Reference Data**: Folder `reference/` containing files like `{judge_agent.name}_1.json`, `{judge_agent.name}_2.json` (SAME Structure).\n\n"

        "## OPERATIONAL PROTOCOL (STRICT SEQUENCE)\n"
        "You must execute the following phases in order. Do not skip steps.\n\n"

        "**PHASE 1: RESEARCH & METRIC DEFINITION (Action: Search & Define Logic)**\n"
        f"1. **RESEARCH**: Use Google Search to find the specific body language habits of {judge_agent.label} (e.g., 'Steve Jobs steeple hand', 'Trump accordion hands').\n"
        "2. **DEFINE METRIC**: Select ONE high-level concept and define the **MATHEMATICAL LOGIC**.\n"
        "   - *Example*: 'Calculate the **Angle** of the elbow (points 11-13-15).'\n"
        "3. **SELECT**: List the specific Landmark IDs required.\n\n"

        "**PHASE 2: FEATURE ENGINEERING COMMAND (Action: Instruct Engineer)**\n"
        "1. Direct the 'Feature_Extractor' to write a Python script.\n"
        "2. **RESTRICTION**: **DO NOT WRITE CODE YOURSELF.** You are the Manager. Only give instructions.\n"
        "3. **CRITICAL INSTRUCTIONS FOR THE SCRIPT**:\n"
        "   - **Import**: `import aura_analysis_lib` (It is in the same folder).\n"
        "   - **Define Metric**: Write a SINGLE function `def calculate_metric(frame):` that takes a list of landmarks and returns a float (or None).\n"
        "     - Implement the math defined in Phase 1 (e.g. angle between 11-13-15).\n"
        "   - **Run Analysis**: Call `print(aura_analysis_lib.run_analysis('" + judge_agent.name + "', calculate_metric))` at the end.\n"
        "   - **NO CSV/JSON LOADING**: Do NOT write code to open files. The library handles ALL reading.\n"
        "4. **STOP** speaking immediately after giving the command.\n\n"

        "**PHASE 3: VERDICT & TERMINATION (Action: Analyze)**\n"
        "1. Wait for the JSON output. Compare `user_value` against `ref_min` and `ref_max`.\n"
        "2. Determine the `severity` score (0.0 to 1.0) using this RANGE-BASED RUBRIC:\n\n"
        
        "   --- JUDGMENT RUBRIC (0.0-1.0 Scale) ---\n"
        
        "   **SEVERITY 0.0 - 0.2 (Perfect Match / Strength)**\n"
        "   - **Condition**: User is **INSIDE** the range (`ref_min` <= user <= `ref_max`) AND positioned near the center (Optimal).\n"
        "   - **Verdict**: This is a **STRENGTH**.\n\n"
        
        "   **SEVERITY 0.3 - 0.5 (Acceptable / Minor Polish)**\n"
        "   - **Condition**: User is **INSIDE** the range, but located near the edges (borderline).\n"
        "   - **Verdict**: **PASS**.\n\n"
        
        "   **SEVERITY 0.6 - 0.8 (Noticeable Deviation / Warning)**\n"
        "   - **Condition**: User is **OUTSIDE** the range, but the distance to the edge is **LESS than 1.0x** the `range_span`.\n"
        "     *Formula*: `dist(user, edge) < range_span`\n"
        "   - **Verdict**: **WARNING**.\n\n"
        
        "   **SEVERITY 0.9 - 1.0 (Critical Failure)**\n"
        "   - **Condition**: User is **OUTSIDE** the range, and the distance is **MORE than 1.0x** the `range_span` (or moving in OPPOSITE direction).\n"
        "   - **Verdict**: **CRITICAL**.\n\n"

        "2. **Final Output**: You MUST output a **Single JSON Object** containing the fields below, followed by the termination keyword.\n"
        "   **Required JSON Structure**:\n"
        "   ```json\n"
        "   {\n"
        "     \"suggestion\": \"(Short title/headline of the advice, e.g. 'Good Hand Posture')\",\n"
        "     \"severity\": 0.1,  // Float between 0.0 (Good) and 1.0 (Critical)\n"
        "     \"description\": \"(Full explanation and professional advice based on the data)\",\n"
        "     \"judge\": \"" + judge_agent.label + "\" // The Name of the Judge\n"
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
            
    return extract_json_from_text(final_comment)

def extract_json_from_text(text):
    import json
    import re

    # If the input is already a dict, return it directly
    if isinstance(text, dict):
        return text
    
    # Ensure text is a string
    if not isinstance(text, str):
        text = str(text)
    
    # Strategy 1: Look for markdown code block
    code_block_pattern = r"```json\s*(\{.*?\})\s*```"
    match = re.search(code_block_pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 2: Look for the LAST valid JSON-like object (Agent often outputs JSON at the end)
    # We find all substring that look like json object { ... }
    # Using a naive stack depth approach or regex is risky for nested.
    # We will try to find the last occurring '}' and match it with a preceding '{'
    
    try:
        # Find the last '}'
        end_idx = text.rfind('}')
        if end_idx != -1:
            # Iterate backwards to find the matching opening '{'
            # (Simple approach: try parsing from every preceding '{' until success)
            # This is O(N^2) worst case but N is small (message size).
            
            # Find all indices of '{' before end_idx
            start_indices = [m.start() for m in re.finditer(r'\{', text[:end_idx])]
            
            # Iterate reversed (from closest '{' to '}')
            for start_idx in reversed(start_indices):
                candidate = text[start_idx : end_idx+1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
        
    # Strategy 3: Fallback to the whole text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
        
    # Strategy 4: Fallback generic error
    # Clean text for error msg
    display_text = text[:100].replace('\n', ' ')
    return {
        "suggestion": "Analysis Parsing Failed",
        "severity": 0.0,
        "description": f"Internal Error: Could not extract valid JSON from Agent output. Please check logs.\n\nRaw Snippet: {display_text}...",
        "judge": "System Error"
    }