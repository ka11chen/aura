from autogen_agentchat.messages import TextMessage

async def run_pipeline(
    feature_extractor,
    judges,
    aggregator,
    input_data: str
):
    # Feature Extractor
    fe_result = await feature_extractor.on_messages(
        [TextMessage(content=input_data, source="user")],
        cancellation_token=None
    )

    feature_json = fe_result.chat_message.content

    # Judges
    judge_outputs = []
    for judge in judges:
        res = await judge.on_messages(
            [TextMessage(content=feature_json, source="feature_extractor")],
            cancellation_token=None
        )
        judge_outputs.append(res.chat_message.content)

    # Aggregator
    combined_input = {
        "feature": feature_json,
        "judges": judge_outputs
    }

    final = await aggregator.on_messages(
        [TextMessage(content=str(combined_input), source="system")],
        cancellation_token=None
    )

    return final.chat_message.content
