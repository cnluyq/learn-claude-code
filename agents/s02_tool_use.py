#!/usr/bin/env python3
# Harness: tool dispatch -- expanding what the model can reach.
"""
s02_tool_use.py - Tools

The agent loop from s01 didn't change. We just added tools to the array
and a dispatch map to route calls.

    +----------+      +-------+      +------------------+
    |   User   | ---> |  LLM  | ---> | Tool Dispatch    |
    |  prompt  |      |       |      | {                |
    +----------+      +---+---+      |   bash: run_bash |
                          ^          |   read: run_read |
                          |          |   write: run_wr  |
                          +----------+   edit: run_edit |
                          tool_result| }                |
                                     +------------------+

Key insight: "The loop didn't change at all. I just added tools."
"""

import os
import subprocess
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks. Act, don't explain."

##add_private_codes_begin############################################################################
from pprint import pprint
import json
# 统计用户输入loop次数
input_counter = 0
# 统计针对每次用户输入agent和LLM交互次数
agent_counter = 0

# 全局统计字典
token_stats = {}

def update_token_stats(response):
    """更新 token 使用统计"""
    model = response.model
    usage = response.usage

    # 确保该模型已有统计条目
    if model not in token_stats:
        token_stats[model] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }

    # 累加各个字段，注意 None 值转为 0
    token_stats[model]["input_tokens"] += usage.input_tokens or 0
    token_stats[model]["output_tokens"] += usage.output_tokens or 0
    token_stats[model]["cache_creation_input_tokens"] += usage.cache_creation_input_tokens or 0
    token_stats[model]["cache_read_input_tokens"] += usage.cache_read_input_tokens or 0

def print_token_stats():
    print("=== Token Usage Statistics (total from session start) ===")
    for model, stats in token_stats.items():
        print(f"Model: {model}")
        print(f"  Input tokens: {stats['input_tokens']}")
        print(f"  Output tokens: {stats['output_tokens']}")
        print(f"  Cache creation input tokens: {stats['cache_creation_input_tokens']}")
        print(f"  Cache read input tokens: {stats['cache_read_input_tokens']}")

def serialize_list(list_data):
    serialized = []
    for item in list_data:
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
##add_private_codes_end############################################################################

def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def run_read(path: str, limit: int = None) -> str:
    try:
        text = safe_path(path).read_text()
        lines = text.splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# -- The dispatch map: {tool_name: handler} --
TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}

TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
]


def agent_loop(messages: list):
    global agent_counter
    while True:
        agent_counter += 1
        print("------------------------------------------------------------------------------------------------------------------------")
        print(f"=== user input (loop#{input_counter})::agent action (loop#{agent_counter}) === calling LLM ......")
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        update_token_stats(response)

        print(f"=== user input (loop#{input_counter})::agent action (loop#{agent_counter}) === response: ")
        if hasattr(response, "model_dump"):
            print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))
        else:
            pprint(response, indent=2, width=120)

        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return
        results = []
        for block in response.content:
            if block.type == "tool_use":
                handler = TOOL_HANDLERS.get(block.name)
                output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                #print(f"> {block.name}: {output[:200]}")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
                print("------------------------------------------------------------------------------------------------------------------------")
                print(f"=== user input (loop#{input_counter})::agent action (loop#{agent_counter}) === user_run_tool result: ")
                results_serialized = serialize_list(results)
                print(json.dumps(results_serialized, indent=2, ensure_ascii=False))

        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms02 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        input_counter += 1
        history.append({"role": "user", "content": query})

        print("------------------------------------------------------------------------------------------------------------------------")
        print("------------------------------------------------------------------------------------------------------------------------")
        print(f"<<<<<< user input (loop#{input_counter}) >>>>>>")
        pprint(f"{query}")
        print(f"<<<<<< hist input (loop#{input_counter}) >>>>>>")
        history_serialized = serialize_list(history)
        print(json.dumps(history_serialized, indent=2, ensure_ascii=False))

        agent_counter = 0
        agent_loop(history)

        print("------------------------------------------------------------------------------------------------------------------------")
        print(f"<<<<<< hist output (loop#{input_counter}) >>>>>>")
        history_serialized = serialize_list(history)
        print(json.dumps(history_serialized, indent=2, ensure_ascii=False))
        print("------------------------------------------------------------------------------------------------------------------------")
        print_token_stats()
        print("------------------------------------------------------------------------------------------------------------------------")
        print("------------------------------------------------------------------------------------------------------------------------")
        #response_content = history[-1]["content"]
        #if isinstance(response_content, list):
        #    for block in response_content:
        #        if hasattr(block, "text"):
        #            print(block.text)
        #print()
