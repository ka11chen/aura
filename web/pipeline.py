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

    judge_results = dict(results_list)

    formatted_output = parse_judgement_results(judge_results)

    print("=====Results=====")
    print(formatted_output)

    return judge_results

def parse_judgement_results(raw_data):
    formatted_output = []

    for judge_key, data in raw_data.items():
        try:
            # If data is already a dict, use it directly
            if isinstance(data, dict):
                formatted_output.append(data)
                continue
            
            # Fallback for string input (legacy support)
            clean_name = judge_key.replace("Judge_", "").replace("_", " ")

            match = re.search(r"(\{.*\})", data, re.DOTALL)

            if match:
                json_part = match.group(1)
                data_dict = json.loads(json_part)
                # ... legacy mapping if needed, but we assume run_judge did its job ...
                formatted_output.append(data_dict)
            else:
                print(f"Warning: No valid JSON found for {judge_key}")

        except json.JSONDecodeError as e:
            print(f"Error parsing JSON for {judge_key}: {e}")
        except Exception as e:
            print(f"Unexpected error for {judge_key}: {e}")

    return formatted_output