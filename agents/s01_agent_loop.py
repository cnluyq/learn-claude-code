#!/usr/bin/env python3
# Harness: the loop -- the model's first connection to the real world.
"""
s01_agent_loop.py - The Agent Loop

The entire secret of an AI coding agent in one pattern:

    while stop_reason == "tool_use":
        response = LLM(messages, tools)
        execute tools
        append results

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> |  Tool   |
    |  prompt  |      |       |      | execute |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                          (loop continues)

This is the core loop: feed tool results back to the model
until the model decides to stop. Production agents layer
policy, hooks, and lifecycle controls on top.
"""

import os
import subprocess
import json
from pprint import pprint
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

anthropic_client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

TOOLS = [{
    "name": "bash",
    "description": "Run a shell command.",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}]

input_counter = 0 
agent_counter = 0

def serialize_history(history):
    serialized = []
    for item in history:
        # 复制一份，避免修改原数据
        item_copy = item.copy()
        # 如果 content 是列表，递归处理其中的对象
        if isinstance(item_copy.get("content"), list):
            new_content = []
            for block in item_copy["content"]:
                if hasattr(block, "model_dump"):
                    new_content.append(block.model_dump())
                elif hasattr(block, "to_dict"):
                    new_content.append(block.to_dict())
                else:
                    new_content.append(block)
            item_copy["content"] = new_content
        serialized.append(item_copy)
    return serialized

def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


# -- The core pattern: a while loop that calls tools until the model stops --
def agent_loop(messages: list):
    global agent_counter
    while True:
        agent_counter += 1
        print(f"=== input #{input_counter}::agent #{agent_counter} === calling LLM ......")
        response = anthropic_client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        
        print(f"=== input #{input_counter}::agent #{agent_counter} === response: ")
        if hasattr(response, "model_dump"):
            print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))
        else:
            pprint(response, indent=2, width=120)
        print("-----------------------------------------------------------------------")

        # Append assistant turn
        messages.append({"role": "assistant", "content": response.content})
        # If the model didn't call a tool, we're done
        if response.stop_reason != "tool_use":
            return
        # Execute each tool call, collect results
        results = []
        for block in response.content:
            if block.type == "tool_use":
                #print(f"\033[33m$ {block.input['command']}\033[0m")
                output = run_bash(block.input["command"])
                #print(output[:200])
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": output})
                print(f"=== input #{input_counter}::agent #{agent_counter} === user_run_tool result: ")
                pprint(results, indent=2, width=120)
                #results_serialized = serialize_history(results)
                #print(json.dumps(results_serialized, indent=2, ensure_ascii=False))
                print("-----------------------------------------------------------------------")

        messages.append({"role": "user_run_tool", "content": results})

if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        input_counter += 1
        history.append({"role": "user", "content": query})
        print(f"<<<<<< input #{input_counter} >>>>>> history_input: ")
        #pprint(history, indent=2, width=12000)
        history_serialized = serialize_history(history)
        print(json.dumps(history_serialized, indent=2, ensure_ascii=False))
        print("-----------------------------------------------------------------------")

        agent_counter = 0
        agent_loop(history)

        print(f"<<<<<< input #{input_counter} >>>>>> history_output: ")
        history_serialized = serialize_history(history)
        print(json.dumps(history_serialized, indent=2, ensure_ascii=False))
        print("-----------------------------------------------------------------------")
        #response_content = history[-1]["content"]
        #if isinstance(response_content, list):
        #    for block in response_content:
        #        if hasattr(block, "text"):
        #            print(block.text)
        #print()
