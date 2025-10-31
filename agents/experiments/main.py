import os
import json
import sys
from pathlib import Path
from pydantic import SecretStr
from dotenv import load_dotenv
load_dotenv('.env')

# Ensure absolute imports like `repo_level_vuldetection.data_processing.*` work when
# running this nested script directly. We add the parent directory that contains
# the `repo_level_vuldetection/` folder to sys.path.
def _ensure_repo_parent_on_sys_path(pkg_name: str = "repo_level_vuldetection") -> None:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / pkg_name).exists():
            repo_parent = p
            if str(repo_parent) not in sys.path:
                sys.path.insert(0, str(repo_parent))
            break

_ensure_repo_parent_on_sys_path()

from prompts import instructions
from repo_level_vuldetection.data_processing.load_data import load_ReposVul_dataset, get_ReposVul_dataitem_vuln_func, get_ReposVul_dataitem_commit_hash

from openhands.sdk import LLM, Conversation
from openhands.tools.preset.default import get_default_agent
from openhands.sdk.event import MessageEvent, ObservationEvent
from openhands.sdk.tool.builtins.finish import FinishObservation
from openhands.sdk.llm import content_to_str


def _extract_final_answer() -> str:
    # Prefer an explicit finish tool observation if present
    for e in reversed(conversation.state.events):
        if isinstance(e, ObservationEvent) and isinstance(e.observation, FinishObservation):
            return e.observation.message.strip()
    # Otherwise, fall back to the last assistant message
    for e in reversed(conversation.state.events):
        if isinstance(e, MessageEvent) and getattr(e, "source", None) == "agent":
            parts = content_to_str(e.llm_message.content)
            if parts:
                return "".join(parts).strip()
    return ""

#API key
api_key = os.getenv("OPENHANDS_API_KEY")
assert api_key is not None, "OPENHANDS_API_KEY environment variable is not set."

# Configure LLM and agent
llm = LLM(
    model="openhands/claude-sonnet-4-5-20250929",
    api_key=SecretStr(api_key),
    service_id="agent",
    drop_params=True,
)
agent = get_default_agent(llm=llm, cli_mode=True)

# Load data
dataset = load_ReposVul_dataset(start_idx=0, end_idx=2)
for item in dataset:
    path_to_repo = item["local_repo_path"]
    func = get_ReposVul_dataitem_vuln_func(item)[0]["function"]
    commit_hash = get_ReposVul_dataitem_commit_hash(item)

    # Start a conversation and set up streaming event logging to a file
    workspace = "/raid/data/ptnguyen/repo_level_vuldetection/agents/openhands_agent-sdk/experiments/"
    log_file = os.path.join("..", "output", f"log_{item['project'].replace('/', '_')}_{commit_hash}.jsonl")

    def make_event_logger(log_path: str):
        def on_event(e):
            from openhands.sdk.event import ActionEvent, ObservationEvent
            record = None
            if isinstance(e, ActionEvent):
                # Serialize the tool call (action) details
                try:
                    args = e.action.model_dump(exclude_none=True) if e.action else None
                except Exception:
                    args = None
                record = {
                    "type": "action",
                    "tool": getattr(e, "tool_name", None),
                    "tool_call_id": getattr(e, "tool_call_id", None),
                    "llm_response_id": getattr(e, "llm_response_id", None),
                    "security_risk": str(getattr(e, "security_risk", "")) or None,
                    "arguments": args,
                }
            elif isinstance(e, ObservationEvent):
                # Serialize the observation/result details
                obs_text = ""
                try:
                    # Prefer LLN-friendly content if available
                    obs_text = "".join(content_to_str(e.observation.to_llm_content))
                except Exception:
                    # Fallback to known fields (e.g., FinishObservation.message)
                    if hasattr(e.observation, "message"):
                        obs_text = getattr(e.observation, "message", "")
                record = {
                    "type": "observation",
                    "tool": getattr(e, "tool_name", None),
                    "tool_call_id": getattr(e, "tool_call_id", None),
                    "action_id": str(getattr(e, "action_id", "")),
                    "result": obs_text,
                }

            if record is not None:
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return on_event

    conversation = Conversation(
        agent=agent,
        workspace=path_to_repo,
        callbacks=[make_event_logger(log_file)],
    )

    # Send a message and let the agent run
    conversation.send_message("".join(instructions).format(path_to_repo=path_to_repo, function_source_code=func, commit_hash=commit_hash))
    conversation.run()

    answer_text = _extract_final_answer()
    output_file = os.path.join("..", "output", f"answer_{item['project'].replace('/', '_')}_{commit_hash}.txt")
    # Optionally write the final answer now
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(answer_text)
    print(f"Saved answer -> {output_file}\nEvent log -> {log_file}")