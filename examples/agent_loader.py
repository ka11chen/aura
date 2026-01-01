import json
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.model_context import UnboundedChatCompletionContext

from autogen_ext.tools.code_execution import PythonCodeExecutionTool
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor

from dotenv import load_dotenv

load_dotenv()

def load_agent_from_json(path: str) -> AssistantAgent:
    with open(path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    cfg = spec["config"]

    # Model client
    model_cfg = cfg["model_client"]["config"]
    model_client = OpenAIChatCompletionClient(
        model=model_cfg["model"],
        api_key=os.getenv("OPENAI_API_KEY")
    )

    print(os.getenv("OPENAI_API_KEY"))

    model_context = UnboundedChatCompletionContext()

    # Tools
    tools = []
    for tool_spec in cfg.get("tools", []):
        if "code_execution" in tool_spec["provider"]:
            exec_cfg = tool_spec["config"]["executor"]["config"]

            executor = LocalCommandLineCodeExecutor(
                timeout=exec_cfg.get("timeout", 300),
                work_dir=exec_cfg.get("work_dir", ".coding"),
                functions_module=exec_cfg.get("functions_module")
            )

            tools.append(
                PythonCodeExecutionTool(
                    executor=executor
                )
            )

    # Agent
    agent = AssistantAgent(
        name=cfg["name"],
        description=cfg.get("description"),
        system_message=cfg["system_message"],
        model_client=model_client,
        model_context=model_context,
        tools=tools,
        model_client_stream=cfg.get("model_client_stream", False),
        reflect_on_tool_use=cfg.get("reflect_on_tool_use", False),
        tool_call_summary_format=cfg.get("tool_call_summary_format", "{result}")
    )

    return agent
