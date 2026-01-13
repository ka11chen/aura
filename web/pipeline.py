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

    final = await aggregator.on_messages(
        [TextMessage(content=str(judge_results), source="system")],
        cancellation_token=None
    )

    print("=====Final=====")
    print(final)

    # formatted_output = parse_judgement_results(judge_results)
    #
    # print("=====Results=====")
    # print(formatted_output)

    return final

def parse_judgement_results(raw_data):
    formatted_output = []

    for judge_key, result_str in raw_data.items():
        try:
            clean_name = judge_key.replace("Judge_", "").replace("_", " ")

            match = re.search(r"(\{.*\})", result_str, re.DOTALL)

            if match:
                json_part = match.group(1)
                data = json.loads(json_part)

                entry = {
                    "suggestion": data.get("metric_analyzed", "General Feedback"),

                    "severity": data.get("severity", 0),

                    "description": f"{clean_name}: {data.get('suggestion', '')}",

                    "judge": clean_name
                }

                formatted_output.append(entry)
            else:
                print(f"Warning: No valid JSON found for {judge_key}")

        except json.JSONDecodeError as e:
            print(f"Error parsing JSON for {judge_key}: {e}")
        except Exception as e:
            print(f"Unexpected error for {judge_key}: {e}")

    return formatted_output