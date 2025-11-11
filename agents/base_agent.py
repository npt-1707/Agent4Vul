from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional
import json
import os


@dataclass
class AgentRunResult:
    """Standard result object for any platform agent run.

    Fields are intentionally generic so subclasses (OpenHands, Claude, Cursor, etc.)
    can populate them consistently.
    """

    # Core outputs
    answer: str = ""              # Final user-facing answer text
    usage: Optional[str] = None    # Human-friendly usage/cost line (if available)

    # Artifacts/paths
    answer_path: Optional[str] = None  # Where the answer was written, if saved
    log_path: Optional[str] = None     # Where event log was written, if saved

    # Optional IDs/metadata (platform-dependent)
    conversation_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Abstract base class for any platform agent.

    Subclass this for specific providers:
      - OpenHandsAgent(BaseAgent)
      - ClaudeAgent(BaseAgent)
      - CursorAgent(BaseAgent)

    Contract (must-have for subclasses):
      - Implement `run()` to execute the agent given a prompt and optional context.
      - Return an `AgentRunResult` with at least `answer` populated.

    Recommended conventions for subclasses:
      - Support writing a JSONL event log via `log_jsonl_path`.
      - Support writing the final answer to `answer_path`.
      - Populate a compact usage string if the platform exposes token/cost data.
    """

    # Minimal, generic configuration shared by many platforms
    model_name: str
    api_key: Optional[str] = None
    service_id: str = "agent"  # logical name for metrics or tracing

    # Arbitrary config hooks (subclasses can read these)
    config: Mapping[str, Any] | None = None

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        service_id: str = "agent",
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENHANDS_API_KEY") or os.getenv("LLM_API_KEY")
        self.service_id = service_id
        self.config = config

    # -----------------------------
    # Abstracts to implement
    # -----------------------------
    @abstractmethod
    def run(
        self,
        prompt: str,
        *,
        workspace: str | Path | None = None,
        callbacks: Optional[Iterable[Callable[[Any], None]]] = None,
        log_jsonl_path: Optional[str | Path] = None,
        answer_path: Optional[str | Path] = None,
        **kwargs: Any,
    ) -> AgentRunResult:
        """Execute the agent with the given prompt.

        Args:
            prompt: User instruction or task prompt.
            workspace: Optional working directory/repo path (platform-dependent).
            callbacks: Optional iterable of event callbacks.
            log_jsonl_path: If provided, append structured events to this JSONL file.
            answer_path: If provided, write the final answer text to this path.
            **kwargs: Extra, platform-specific parameters.

        Returns:
            AgentRunResult with answer, usage, and optional metadata.
        """
        raise NotImplementedError

    # -----------------------------
    # Reusable helpers for subclasses
    # -----------------------------
    @staticmethod
    def save_text(path: str | Path, content: str) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return str(path)

    @staticmethod
    def make_jsonl_logger(
        log_path: str | Path,
        *,
        serializer: Optional[Callable[[Any], Optional[dict[str, Any]]]] = None,
    ) -> Callable[[Any], None]:
        """Create a JSONL logger callback for arbitrary events.

        - serializer(event) should return a JSON-serializable dict or None to skip.
        - If no serializer is provided, event is written as-is if it's a mapping.
        """
        out_path = Path(log_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        def _default_serializer(event: Any) -> Optional[dict[str, Any]]:
            if isinstance(event, dict):
                return event  # naive passthrough
            # Not serializable -> skip
            return None

        _ser = serializer or _default_serializer

        def _callback(event: Any) -> None:
            try:
                record = _ser(event)
                if record is None:
                    return
                with out_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception:
                # Best-effort logger: ignore serialization errors
                pass

        return _callback

    @staticmethod
    def format_usage_line(
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cache_read_tokens: int | None = None,
        prompt_tokens: int | None = None,
        reasoning_tokens: int | None = None,
        total_cost_usd: float | None = None,
    ) -> str:
        """Build a compact, human-friendly usage line.

        Example: "Tokens: ↑ input 37.79K • cache hit 44.85% • reasoning 1.67K • ↓ output 3.98K • $ 0.2857"
        """
        def _abbr(n: Optional[int]) -> str:
            if n is None:
                return "0"
            v = int(max(0, n))
            if v >= 1_000_000_000:
                s = f"{v / 1_000_000_000:.2f}B"
            elif v >= 1_000_000:
                s = f"{v / 1_000_000:.2f}M"
            elif v >= 1_000:
                s = f"{v / 1_000:.2f}K"
            else:
                return str(v)
            return s.replace(".0", "")

        parts: list[str] = []
        parts.append(f"↑ input {_abbr(input_tokens)}")

        # Cache hit rate if prompt_tokens available
        if prompt_tokens and prompt_tokens > 0 and (cache_read_tokens or 0) > 0:
            rate = (cache_read_tokens or 0) / prompt_tokens * 100.0
            parts.append(f"cache hit {rate:.2f}%")

        if reasoning_tokens and reasoning_tokens > 0:
            parts.append(f"reasoning {_abbr(reasoning_tokens)}")

        parts.append(f"↓ output {_abbr(output_tokens)}")

        cost_str = f"{(total_cost_usd or 0.0):.4f}" if (total_cost_usd or 0.0) > 0 else "0.00"
        parts.append(f"$ {cost_str}")

        return "Tokens: " + " • ".join(parts)
