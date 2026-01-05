from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.messages import TextMessage

async def run_analysis_session(feature_extractor_agent, judge_agent):
    termination = TextMentionTermination("TERMINATE")

    team = RoundRobinGroupChat(
        participants=[judge_agent, feature_extractor_agent],
        termination_condition=termination,
        max_turns=6
    )

    task = (
        f"You are {judge_agent.name}. Please evaluate the speaker based on the data in 'landmarks.json'.\n"
        "You can instruct the Engineer to write Python scripts to calculate any specific features or metrics you prioritize.\n"
        "Once the analysis is complete, provide a summary verdict and end your response with 'TERMINATE'."
    )

    print(f"--- Running Session: {judge_agent.name} ---")

    result = await team.run(task=task)

    print(result)

    final_comment = f"(No comment found for {judge_agent.name})"

    for msg in reversed(result.messages):
        if isinstance(msg, TextMessage) and msg.source == judge_agent.name:
            final_comment = msg.content
            break

    print(f"[{judge_agent.name} Final Verdict]: {final_comment[:100]}...")

    return final_comment.replace("TERMINATE", "").strip()