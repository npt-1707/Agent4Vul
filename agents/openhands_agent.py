from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

from pydantic import SecretStr

# OpenHands SDK
from openhands.sdk import LLM, Conversation
from openhands.tools.preset.default import get_default_agent
from openhands.sdk.event import MessageEvent, ObservationEvent
from openhands.sdk.tool.builtins.finish import FinishObservation
from openhands.sdk.llm import content_to_str


@dataclass
class OpenHandsAgent:
    """A thin abstraction over an OpenHands agent + conversation lifecycle.

    Attributes:
        llm_model_name: Model id passed to OpenHands LLM.
        api_key: API key used by the LLM provider (read from env if None).
        service_id: Logical service id used by the LLM registry/metrics.
        base_url: Optional custom base URL (e.g., proxy or self-hosted endpoint).
        cli_mode: Whether to render CLI-friendly output in default agent.
        drop_params: Passed to LLM (lets SDK normalize provider params).

    Usage:
        agent = Agent(llm_model_name="openhands/claude-sonnet-4-5-20250929")
        result = agent.run(
            prompt="Analyze this repo...",
            workspace="/path/to/workspace-or-repo",
            log_jsonl_path="./output/agent_log.jsonl",
            answer_path="./output/answer.txt",
        )
        print(result["answer"])  # final assistant answer
        print(result["usage"])   # token/cost summary if available
    """

    llm_model_name: str = "openhands/claude-sonnet-4-5-20250929"
    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENHANDS_API_KEY") or os.getenv("LLM_API_KEY")
    )
    service_id: str = "agent"
    base_url: Optional[str] = None
    cli_mode: bool = True
    drop_params: bool = True

    # Advanced toggles (extend as needed)
    native_tool_calling: Optional[bool] = None

    def _build_llm(self) -> LLM:
        if not self.api_key:
            raise ValueError("Missing API key. Set OPENHANDS_API_KEY or provide api_key.")
        return LLM(
            model=self.llm_model_name,
            api_key=SecretStr(self.api_key),
            service_id=self.service_id,
            base_url=self.base_url,
            drop_params=self.drop_params,
            native_tool_calling=self.native_tool_calling,
        )

    @staticmethod
    def _extract_final_answer(conversation: Conversation) -> str:
        # Prefer a finish observation (explicit final answer)
        for e in reversed(conversation.state.events):
            if isinstance(e, ObservationEvent) and isinstance(e.observation, FinishObservation):
                return e.observation.message.strip()
        # Fallback to the last assistant message
        for e in reversed(conversation.state.events):
            if isinstance(e, MessageEvent) and getattr(e, "source", None) == "agent":
                parts = content_to_str(e.llm_message.content)
                if parts:
                    return "".join(parts).strip()
        return ""

    @staticmethod
    def _abbr(n: int | float) -> str:
        n = int(n or 0)
        if n >= 1_000_000_000:
            s = f"{n / 1_000_000_000:.2f}B"
        elif n >= 1_000_000:
            s = f"{n / 1_000_000:.2f}M"
        elif n >= 1_000:
            s = f"{n / 1_000:.2f}K"
        else:
            return str(n)
        return s.replace(".0", "")

    @classmethod
    def _format_usage(cls, conversation: Conversation) -> str:
        # Combine metrics across services (if any)
        try:
            metrics = conversation.conversation_stats.get_combined_metrics()
        except Exception:
            return ""
        usage = getattr(metrics, "accumulated_token_usage", None)
        if not usage:
            return ""
        input_tokens = cls._abbr(usage.prompt_tokens or 0)
        output_tokens = cls._abbr(usage.completion_tokens or 0)
        prompt = usage.prompt_tokens or 0
        cache_read = usage.cache_read_tokens or 0
        cache_rate = f"{(cache_read / prompt * 100):.2f}%" if prompt > 0 else "N/A"
        reasoning_tokens = usage.reasoning_tokens or 0
        cost = getattr(metrics, "accumulated_cost", 0.0) or 0.0
        cost_str = f"{cost:.4f}" if cost > 0 else "0.00"
        parts: list[str] = [
            f"↑ input {input_tokens}",
            f"cache hit {cache_rate}",
        ]
        if reasoning_tokens > 0:
            parts.append(f"reasoning {cls._abbr(reasoning_tokens)}")
        parts.append(f"↓ output {output_tokens}")
        parts.append(f"$ {cost_str}")
        return "Tokens: " + " • ".join(parts)

    @staticmethod
    def _make_event_logger(log_path: str) -> Callable:
        """Create a JSONL event logger for actions and observations."""
        def on_event(e):
            from openhands.sdk.event import ActionEvent, ObservationEvent, MessageEvent, SystemPromptEvent
            from openhands.sdk.event.condenser import Condensation
            record = None
            if isinstance(e, ActionEvent):
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
                obs_text = ""
                try:
                    obs_text = "".join(content_to_str(e.observation.to_llm_content))
                except Exception:
                    if hasattr(e.observation, "message"):
                        obs_text = getattr(e.observation, "message", "")
                record = {
                    "type": "observation",
                    "tool": getattr(e, "tool_name", None),
                    "tool_call_id": getattr(e, "tool_call_id", None),
                    "action_id": str(getattr(e, "action_id", "")),
                    "result": obs_text,
                }
            elif isinstance(e, MessageEvent) and getattr(e, "source", None) == "agent":
                # Optional: record assistant messages (compact)
                parts = content_to_str(e.llm_message.content)
                record = {
                    "type": "assistant_message",
                    "content": "".join(parts) if parts else "",
                }
            elif isinstance(e, SystemPromptEvent):
                record = {
                    "type": "system_prompt",
                    "content": "".join(content_to_str(e.system_prompt.content)),
                }
            elif isinstance(e, Condensation):
                record = {
                    "type": "condensation",
                    "content": "".join(content_to_str(e.condensed_message.content)),
                }
            else:
                print(type(e))
            if record is not None:
                Path(log_path).parent.mkdir(parents=True, exist_ok=True)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return on_event
    

    def run(
        self,
        prompt: str,
        workspace: str | Path,
        *,
        callbacks: Optional[Iterable[Callable]] = None,
        log_jsonl_path: Optional[str | Path] = None,
        answer_path: Optional[str | Path] = None,
    ) -> dict:
        """Run the agent on a prompt in the given workspace.

        Args:
            prompt: User instruction or task prompt to send to the agent.
            workspace: Path to working directory (repo or project root) for tools.
            callbacks: Optional iterable of event callbacks to attach.
            log_jsonl_path: If provided, stream events to this JSONL file.
            answer_path: If provided, write the final answer to this file.

        Returns:
            dict with keys: answer, usage, conversation_id, log_path, answer_path
        """
        llm = self._build_llm()
        core_agent = get_default_agent(llm=llm, cli_mode=self.cli_mode)

        cb_list: list[Callable] = list(callbacks) if callbacks else []
        if log_jsonl_path is not None:
            cb_list.append(self._make_event_logger(str(log_jsonl_path)))

        conv = Conversation(
            agent=core_agent,
            workspace=str(workspace),
            callbacks=cb_list if cb_list else None,
        )

        conv.send_message(prompt)
        conv.run()

        answer = self._extract_final_answer(conv)
        usage = self._format_usage(conv)

        if answer_path is not None:
            Path(answer_path).parent.mkdir(parents=True, exist_ok=True)
            with open(answer_path, "w", encoding="utf-8") as f:
                f.write(answer)

        result = {
            "answer": answer,
            "usage": usage,
            "conversation_id": str(conv.id),
            "log_path": str(log_jsonl_path) if log_jsonl_path else None,
            "answer_path": str(answer_path) if answer_path else None,
        }
        return result
