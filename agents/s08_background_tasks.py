#!/usr/bin/env python3
# Harness: background execution -- the model thinks while the harness waits.
"""
s08_background_tasks.py - Background Tasks

Run commands in background threads. A notification queue is drained
before each LLM call to deliver results.

    Main thread                Background thread
    +-----------------+        +-----------------+
    | agent loop      |        | task executes   |
    | ...             |        | ...             |
    | [LLM call] <---+------- | enqueue(result) |
    |  ^drain queue   |        +-----------------+
    +-----------------+

    Timeline:
    Agent ----[spawn A]----[spawn B]----[other work]----
                 |              |
                 v              v
              [A runs]      [B runs]        (parallel)
                 |              |
                 +-- notification queue --> [results injected]

Key insight: "Fire and forget -- the agent doesn't block while the command runs."
"""

import os
import subprocess
import threading
import uuid
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

SYSTEM = f"You are a coding agent at {WORKDIR}. Use background_run for long-running commands."

##add_private_codes_begin############################################################################
from pprint import pprint
import json
import time
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
        item_copy = item.copy()
        content = item_copy.get("content")
        if isinstance(content, list):
            new_content = []
            for block in content:
                # 将 block 转换为 dict
                if hasattr(block, "model_dump"):
                    block_dict = block.model_dump()
                elif hasattr(block, "to_dict"):
                    block_dict = block.to_dict()
                else:
                    block_dict = block
                # 处理 tool_result 的 content
                if isinstance(block_dict, dict) and block_dict.get("type") == "tool_result":
                    block_content = block_dict.get("content")
                    if isinstance(block_content, str):
                        try:
                            block_dict["content"] = json.loads(block_content)
                        except json.JSONDecodeError:
                            pass
                new_content.append(block_dict)
            item_copy["content"] = new_content
        elif isinstance(content, str):
            # 如果是字符串，尝试解析为 JSON（适用于顶层 tool_result）
            try:
                parsed = json.loads(content)
                item_copy["content"] = parsed
            except json.JSONDecodeError:
                pass  # 保持原样
        serialized.append(item_copy)
    return serialized
##add_private_codes_end############################################################################

# -- BackgroundManager: threaded execution + notification queue --
class BackgroundManager:
    def __init__(self):
        self.tasks = {}  # task_id -> {status, result, command}
        self._notification_queue = []  # completed task results
        self._lock = threading.Lock()

    def run(self, command: str) -> str:
        """Start a background thread, return task_id immediately."""
        task_id = str(uuid.uuid4())[:8]
        self.tasks[task_id] = {"status": "running", "result": None, "command": command}
        thread = threading.Thread(
            target=self._execute, args=(task_id, command), daemon=True
        )
        thread.start()
        return f"Background task {task_id} started: {command[:80]}"

    def _execute(self, task_id: str, command: str):
        """Thread target: run subprocess, capture output, push to queue."""
        try:
            r = subprocess.run(
                command, shell=True, cwd=WORKDIR,
                capture_output=True, text=True, timeout=300
            )
            output = (r.stdout + r.stderr).strip()[:50000]
            status = "completed"
        except subprocess.TimeoutExpired:
            output = "Error: Timeout (300s)"
            status = "timeout"
        except Exception as e:
            output = f"Error: {e}"
            status = "error"
        self.tasks[task_id]["status"] = status
        self.tasks[task_id]["result"] = output or "(no output)"
        with self._lock:
            self._notification_queue.append({
                "task_id": task_id,
                "status": status,
                "command": command[:80],
                "result": (output or "(no output)")[:500],
            })

    def check(self, task_id: str = None) -> str:
        """Check status of one task or list all."""
        if task_id:
            t = self.tasks.get(task_id)
            if not t:
                return f"Error: Unknown task {task_id}"
            return f"[{t['status']}] {t['command'][:60]}\n{t.get('result') or '(running)'}"
        lines = []
        for tid, t in self.tasks.items():
            lines.append(f"{tid}: [{t['status']}] {t['command'][:60]}")
        return "\n".join(lines) if lines else "No background tasks."

    def drain_notifications(self) -> list:
        """Return and clear all pending completion notifications."""
        with self._lock:
            notifs = list(self._notification_queue)
            self._notification_queue.clear()
        return notifs


BG = BackgroundManager()


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
        c = fp.read_text()
        if old_text not in c:
            return f"Error: Text not found in {path}"
        fp.write_text(c.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


TOOL_HANDLERS = {
    "bash":             lambda **kw: run_bash(kw["command"]),
    "read_file":        lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file":       lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":        lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "background_run":   lambda **kw: BG.run(kw["command"]),
    "check_background": lambda **kw: BG.check(kw.get("task_id")),
}

TOOLS = [
    {"name": "bash", "description": "Run a shell command (blocking).",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "background_run", "description": "Run command in background thread. Returns task_id immediately.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "check_background", "description": "Check background task status. Omit task_id to list all.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "string"}}}},
]


def agent_loop(messages: list):
    global agent_counter
    while True:
        agent_counter += 1
        print("------------------------------------------------------------------------------------------------------------------------")
        print(f"=== user#{input_counter}::agent#{agent_counter} === {time.strftime('%Y-%m-%d %H:%M:%S')} calling LLM ......")

        # Drain background notifications and inject as system message before LLM call
        notifs = BG.drain_notifications()
        if notifs and messages:
            notif_text = "\n".join(
                f"[bg:{n['task_id']}] {n['status']}: {n['result']}" for n in notifs
            )
            messages.append({"role": "user", "content": f"<background-results>\n{notif_text}\n</background-results>"})
            messages.append({"role": "assistant", "content": "Noted background results."})
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        update_token_stats(response)

        print(f"=== user#{input_counter}::agent#{agent_counter} === {time.strftime('%Y-%m-%d %H:%M:%S')} LLM response: ")
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
                try:
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                except Exception as e:
                    output = f"Error: {e}"
                #print(f"> {block.name}: {str(output)[:200]}")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})

        print("------------------------------------------------------------------------------------------------------------------------")
        print(f"=== user#{input_counter}::agent#{agent_counter} === {time.strftime('%Y-%m-%d %H:%M:%S')} user_run_tool \"{block.name}\" result: ")
        results_serialized = serialize_list(results)
        print(json.dumps(results_serialized, indent=2, ensure_ascii=False))

        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms08 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})

##add_private_codes_begin############################################################################
        input_counter += 1
        print("------------------------------------------------------------------------------------------------------------------------")
        print("------------------------------------------------------------------------------------------------------------------------")
        print(f"<<<<<< user input (loop#{input_counter}) {time.strftime('%Y-%m-%d %H:%M:%S')} >>>>>>")
        pprint(f"{query}")
        print(f"<<<<<< hist input (loop#{input_counter}) {time.strftime('%Y-%m-%d %H:%M:%S')} >>>>>>")
        history_serialized = serialize_list(history)
        print(json.dumps(history_serialized, indent=2, ensure_ascii=False))
        agent_counter = 0
##add_private_codes_end############################################################################

        agent_loop(history)

##add_private_codes_begin############################################################################
        print("------------------------------------------------------------------------------------------------------------------------")
        print(f"<<<<<< hist output (loop#{input_counter}) {time.strftime('%Y-%m-%d %H:%M:%S')} >>>>>>")
        history_serialized = serialize_list(history)
        print(json.dumps(history_serialized, indent=2, ensure_ascii=False))
        print("------------------------------------------------------------------------------------------------------------------------")
        print_token_stats()
        print("------------------------------------------------------------------------------------------------------------------------")
        print("------------------------------------------------------------------------------------------------------------------------")
##add_private_codes_end############################################################################

        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()
