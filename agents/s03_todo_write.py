#!/usr/bin/env python3
# Harness: planning -- keeping the model on course without scripting the route.
"""
s03_todo_write.py - TodoWrite

The model tracks its own progress via a TodoManager. A nag reminder
forces it to keep updating when it forgets.

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> | Tools   |
    |  prompt  |      |       |      | + todo  |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                                |
                    +-----------+-----------+
                    | TodoManager state     |
                    | [ ] task A            |
                    | [>] task B <- doing   |
                    | [x] task C            |
                    +-----------------------+
                                |
                    if rounds_since_todo >= 3:
                      inject <reminder>

Key insight: "The agent can track its own progress -- and I can see it."
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

SYSTEM = f"""You are a coding agent at {WORKDIR}.
Use the todo tool to plan multi-step tasks. Mark in_progress before starting, completed when done.
Prefer tools over prose."""

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

# -- TodoManager: structured state the LLM writes to --
class TodoManager:
    def __init__(self):
        self.items = []

    def update(self, items: list) -> str:
        if len(items) > 20:
            raise ValueError("Max 20 todos allowed")
        validated = []
        in_progress_count = 0
        for i, item in enumerate(items):
            text = str(item.get("text", "")).strip()
            status = str(item.get("status", "pending")).lower()
            item_id = str(item.get("id", str(i + 1)))
            if not text:
                raise ValueError(f"Item {item_id}: text required")
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Item {item_id}: invalid status '{status}'")
            if status == "in_progress":
                in_progress_count += 1
            validated.append({"id": item_id, "text": text, "status": status})
        if in_progress_count > 1:
            raise ValueError("Only one task can be in_progress at a time")
        self.items = validated
        return self.render()

    def render(self) -> str:
        if not self.items:
            return "No todos."
        lines = []
        for item in self.items:
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}[item["status"]]
            lines.append(f"{marker} #{item['id']}: {item['text']}")
        done = sum(1 for t in self.items if t["status"] == "completed")
        lines.append(f"\n({done}/{len(self.items)} completed)")
        return "\n".join(lines)


TODO = TodoManager()


# -- Tool implementations --
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
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"

def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
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


TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "todo":       lambda **kw: TODO.update(kw["items"]),
}

# 定义所有告知LLM可用的工具列表。这个非常重要，确定了LLM除了对话外，可以用来指示做动作的能力范围
TOOLS = [
  {
    "name": "bash",
    "description": "Run a shell command.",
    "input_schema": {
      "type": "object",
      "properties": {
        "command": {
          "type": "string"
        }
      },
      "required": [
        "command"
      ]
    }
  },
  {
    "name": "read_file",
    "description": "Read file contents.",
    "input_schema": {
      "type": "object",
      "properties": {
        "path": {
          "type": "string"
        },
        "limit": {
          "type": "integer"
        }
      },
      "required": [
        "path"
      ]
    }
  },
  {
    "name": "write_file",
    "description": "Write content to file.",
    "input_schema": {
      "type": "object",
      "properties": {
        "path": {
          "type": "string"
        },
        "content": {
          "type": "string"
        }
      },
      "required": [
        "path",
        "content"
      ]
    }
  },
  {
    "name": "edit_file",
    "description": "Replace exact text in file.",
    "input_schema": {
      "type": "object",
      "properties": {
        "path": {
          "type": "string"
        },
        "old_text": {
          "type": "string"
        },
        "new_text": {
          "type": "string"
        }
      },
      "required": [
        "path",
        "old_text",
        "new_text"
      ]
    }
  },
  {
    "name": "todo",
    "description": "Update task list. Track progress on multi-step tasks.",
    "input_schema": {
      "type": "object",
      "properties": {
        "items": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "id": {
                "type": "string"
              },
              "text": {
                "type": "string"
              },
              "status": {
                "type": "string",
                "enum": [
                  "pending",
                  "in_progress",
                  "completed"
                ]
              }
            },
            "required": [
              "id",
              "text",
              "status"
            ]
          }
        }
      },
      "required": [
        "items"
      ]
    }
  },
]

# -- Agent loop with nag reminder injection --
def agent_loop(messages: list):
    global agent_counter
    rounds_since_todo = 0
    while True:
        agent_counter += 1
        print("------------------------------------------------------------------------------------------------------------------------")
        print(f"=== user input (loop#{input_counter})::agent action (loop#{agent_counter}) === calling LLM ......")

        # Nag reminder is injected below, alongside tool results
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
        used_todo = False
        for block in response.content:
            if block.type == "tool_use":
                handler = TOOL_HANDLERS.get(block.name)
                try:
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                except Exception as e:
                    output = f"Error: {e}"
                #print(f"> {block.name}: {str(output)[:200]}")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})

                if block.name == "todo":
                    used_todo = True
        rounds_since_todo = 0 if used_todo else rounds_since_todo + 1
        if rounds_since_todo >= 3:
            results.insert(0, {"type": "text", "text": "<reminder>Update your todos.</reminder>"})

        print("------------------------------------------------------------------------------------------------------------------------")
        print(f"=== user input (loop#{input_counter})::agent action (loop#{agent_counter}) === user_run_tool result: ")
        results_serialized = serialize_list(results)
        print(json.dumps(results_serialized, indent=2, ensure_ascii=False))

        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms03 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})

        input_counter += 1
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
