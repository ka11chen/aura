import json
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.model_context import UnboundedChatCompletionContext

from autogen_ext.tools.code_execution import PythonCodeExecutionTool
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor

from dotenv import load_dotenv, find_dotenv

from autogen_core.models import ModelInfo

load_dotenv(find_dotenv())

custom_model_info = ModelInfo(
    vision=False,
    function_calling=True,
    json_output=True,
    family="gpt-4",
    structured_output=True
)

def load_agent_from_json(path: str) -> AssistantAgent:
    with open(path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    cfg = spec["config"]

    model_cfg = cfg["model_client"]["config"]

    api_key = model_cfg.get("api_key") or os.getenv("OPENAI_API_KEY")

    model_client = OpenAIChatCompletionClient(
        model=model_cfg["model"],
        api_key=api_key,
        model_info=custom_model_info
    )

    model_context = UnboundedChatCompletionContext()

    tools = []
    for tool_spec in cfg.get("tools", []):
        if "code_execution" in tool_spec.get("provider", ""):

            tool_config = tool_spec.get("config", {})
            executor_wrapper = tool_config.get("executor", {})
            executor_config = executor_wrapper.get("config", {})

            executor_args = {
                "timeout": executor_config.get("timeout", 300),
                "work_dir": executor_config.get("work_dir", ".coding"),
            }

            func_mod = executor_config.get("functions_module")
            if func_mod:
                executor_args["functions_module"] = func_mod

            executor = LocalCommandLineCodeExecutor(**executor_args)

            tools.append(
                PythonCodeExecutionTool(
                    executor=executor
                )
            )

    agent = AssistantAgent(
        name=cfg["name"],
        description=cfg.get("description", "A helpful AI assistant"),
        system_message=cfg["system_message"],
        model_client=model_client,
        model_context=model_context,
        tools=tools,
        model_client_stream=cfg.get("model_client_stream", False),
        reflect_on_tool_use=cfg.get("reflect_on_tool_use", False),
        tool_call_summary_format=cfg.get("tool_call_summary_format", "{result}")
    )

    return agent