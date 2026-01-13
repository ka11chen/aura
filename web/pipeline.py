import json
import re

from autogen_agentchat.messages import TextMessage
from run_judge import run_analysis_session
import asyncio

async def run_pipeline(
    feature_extractor,
    judges,
    aggregator,
):
    async def process_single_judge(judge_agent):
        print(f"Judge Analysis: {judge_agent.name}...")

        result = await run_analysis_session(
            feature_extractor_agent=feature_extractor,
            judge_agent=judge_agent,
        )

        print(f"{judge_agent.name} Done.")
        return judge_agent.name, result

    tasks = [process_single_judge(judge) for judge in judges]

    results_list = await asyncio.gather(*tasks)

    judge_results = dict(results_list)

    parsed_results = {}
    
    for judge_id, raw_val in judge_results.items():
        parsed = None
        
        try:
            if isinstance(raw_val, dict):
                parsed = raw_val
            else:
                parsed = extract_json_legacy(raw_val)
        except Exception:
            pass
            
        if parsed:
            if 'judge' not in parsed:
                 clean_name = judge_id.replace("Judge_", "").replace("_", " ")
                 parsed['judge'] = clean_name
            
            parsed_results[judge_id] = parsed
        else:
            parsed_results[judge_id] = {
                "metric_analyzed": "Error",
                "severity": 3,
                "suggestion": "Failed to parse analysis result.",
                "judge": judge_id
            }

    formatted_output = list(parsed_results.values())

    print("=====Results=====")
    print(formatted_output)

    final = await aggregator.on_messages(
        [TextMessage(content=str(formatted_output), source="system")],
        cancellation_token=None
    )

    print(final)

    return formatted_output

def extract_json_legacy(text):
    if not isinstance(text, str): return text
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
             return None
    try:
        return json.loads(text)
    except:
        return None
