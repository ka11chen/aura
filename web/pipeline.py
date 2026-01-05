from autogen_agentchat.messages import TextMessage
from run_judge import run_analysis_session

async def run_pipeline(
    feature_extractor,
    judges,
    aggregator,
):
    judge_results = {}

    for judge in judges:
        print(f"Judge Analysis: {judge.name}...")

        result = await run_analysis_session(
            feature_extractor_agent=feature_extractor,
            judge_agent=judge,
        )

        judge_results[judge.name] = result
        print(f"{judge.name} Done.")

    print("=====Judge Results=====")
    print(judge_results)

    final = await aggregator.on_messages(
        [TextMessage(content=str(judge_results), source="system")],
        cancellation_token=None
    )

    print("=====Final=====")
    print(final)

    return final.chat_message.content
