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

    print("=====Results=====")
    print(judge_results)

    final = await aggregator.on_messages(
        [TextMessage(content=str(judge_results), source="system")],
        cancellation_token=None
    )

    final_content = final.chat_message.content

    print(final_content)

    return final_content

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
