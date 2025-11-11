"""
Claude Agent implementation using the Claude Agent SDK.

This module provides a wrapper around the Claude Agent SDK (claude-agent-sdk)
that conforms to the BaseAgent interface for use in the vulnerability detection pipeline.

Installation:
    pip install claude-agent-sdk

Environment Variables:
    ANTHROPIC_API_KEY or CLAUDE_API_KEY - API key for Claude
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from .base_agent import AgentRunResult, BaseAgent

try:
    from claude_agent_sdk import (
        ClaudeSDKClient,
        ClaudeAgentOptions,
        AssistantMessage,
        ResultMessage,
        TextBlock,
        ToolUseBlock,
        ToolResultBlock,
        CLINotFoundError,
        ProcessError,
    )
    CLAUDE_SDK_AVAILABLE = True
except ImportError:
    CLAUDE_SDK_AVAILABLE = False
    ClaudeSDKClient = None
    ClaudeAgentOptions = None


@dataclass
class ClaudeAgent(BaseAgent):
    """Claude Agent implementation using Claude Agent SDK.

    This agent uses the Claude Agent SDK (claude-agent-sdk) which provides
    access to Claude with built-in tool use capabilities for local file operations,
    bash commands, and code analysis. 
    
    IMPORTANT: Web search tools (WebSearch, WebFetch) are explicitly EXCLUDED
    to ensure the agent only analyzes local repository code without external data.

    Attributes:
        model_name: Claude model name (e.g., "claude-3-5-sonnet-20241022", "claude-opus-4-20250514")
        api_key: Anthropic API key (reads from ANTHROPIC_API_KEY or CLAUDE_API_KEY if None)
        service_id: Logical service identifier for tracking
        config: Additional configuration options
        permission_mode: Permission mode for tool usage ("default", "acceptEdits", "bypassPermissions", "plan")
        allowed_tools: List of allowed tool names. DEFAULT: ["Read", "Write", "Edit", "Bash", "Grep", "Glob"]
                      Web tools (WebSearch, WebFetch) are NOT allowed by default.
        system_prompt: System prompt to use (None for default Claude Code prompt)
        max_turns: Maximum conversation turns (None for unlimited)
        cwd: Working directory for the agent (defaults to workspace)

    Allowed Built-in Tools (Local Analysis Only):
        - Read: Read local files with line ranges
        - Write: Write new local files
        - Edit: Edit local files with search/replace
        - Bash: Execute shell commands in workspace
        - Grep: Search local files with regex
        - Glob: Find local files by pattern
        - Task: Delegate to subagents (optional)
        - NotebookEdit: Edit Jupyter notebooks (optional)

    Explicitly EXCLUDED Tools (Web Access):
        - WebSearch: Search the web (DISABLED)
        - WebFetch: Fetch web content (DISABLED)

    Example:
        agent = ClaudeAgent(
            model_name="claude-3-5-sonnet-20241022",
            api_key="sk-ant-...",
            permission_mode="acceptEdits",
            allowed_tools=["Read", "Write", "Bash", "Grep", "Glob"]  # No web tools
        )

        result = agent.run(
            prompt="Analyze this function for vulnerabilities...",
            workspace="/path/to/repo",
            log_jsonl_path="./output/log.jsonl",
            answer_path="./output/answer.txt"
        )

        print(result["answer"])
        print(result["usage"])
    """

    # Claude-specific configuration
    permission_mode: str = "acceptEdits"  # "default", "acceptEdits", "bypassPermissions", "plan"
    
    # Default tools: LOCAL file/code analysis only, NO web access
    allowed_tools: list[str] = field(default_factory=lambda: [
        "Read",    # Read local files
        "Write",   # Write local files
        "Edit",    # Edit local files
        "Bash",    # Execute local commands
        "Grep",    # Search local files
        "Glob",    # Find local files
        # WebSearch and WebFetch are EXPLICITLY EXCLUDED
    ])
    system_prompt: Optional[str] = None
    max_turns: Optional[int] = None
    cwd: Optional[str | Path] = None

    def __post_init__(self):
        """Validate that Claude SDK is available and tools are restricted."""
        if not CLAUDE_SDK_AVAILABLE:
            raise ImportError(
                "claude-agent-sdk is not installed. "
                "Install it with: pip install claude-agent-sdk"
            )

        # Validate that web search tools are not allowed
        prohibited_tools = ["WebSearch", "WebFetch"]
        for tool in prohibited_tools:
            if tool in self.allowed_tools:
                raise ValueError(
                    f"Web tool '{tool}' is not allowed for security reasons. "
                    f"ClaudeAgent must analyze code locally without external web access."
                )

        # Override API key lookup to check Claude-specific env vars
        if not self.api_key:
            self.api_key = (
                os.getenv("ANTHROPIC_API_KEY") or
                os.getenv("CLAUDE_API_KEY") or
                os.getenv("OPENHANDS_API_KEY") or
                os.getenv("LLM_API_KEY")
            )

        if not self.api_key:
            raise ValueError(
                "Missing API key. Set ANTHROPIC_API_KEY, CLAUDE_API_KEY, or provide api_key parameter."
            )

    def _build_options(self, workspace: str | Path | None = None) -> ClaudeAgentOptions:
        """Build ClaudeAgentOptions from current configuration."""
        # Determine working directory
        work_dir = self.cwd or workspace
        if work_dir:
            work_dir = str(Path(work_dir).resolve())

        options = ClaudeAgentOptions(
            allowed_tools=self.allowed_tools,
            permission_mode=self.permission_mode,
            cwd=work_dir,
            max_turns=self.max_turns,
        )

        # Add system prompt if provided
        if self.system_prompt:
            options.system_prompt = self.system_prompt

        # Add model if it's a valid override
        # (Claude SDK accepts "sonnet", "opus", "haiku" or None for default)
        if self.model_name:
            # Map common model names to SDK names
            model_map = {
                "claude-3-5-sonnet": "sonnet",
                "claude-3-opus": "opus",
                "claude-3-haiku": "haiku",
                "sonnet": "sonnet",
                "opus": "opus",
                "haiku": "haiku",
            }
            # Try to extract model type from full name
            for key, value in model_map.items():
                if key in self.model_name.lower():
                    options.model = value
                    break

        return options

    @staticmethod
    def _extract_final_answer(messages: list[Any]) -> str:
        """Extract the final answer text from Claude's messages.

        Args:
            messages: List of messages from Claude SDK

        Returns:
            Final answer text
        """
        answer_parts = []

        for message in messages:
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        answer_parts.append(block.text)

        return "\n".join(answer_parts).strip() if answer_parts else ""

    @staticmethod
    def _format_usage_from_result(result_msg: ResultMessage) -> str:
        """Format usage information from ResultMessage.

        Args:
            result_msg: ResultMessage with usage data

        Returns:
            Formatted usage string
        """
        if not result_msg or not hasattr(result_msg, 'usage'):
            return ""

        usage = result_msg.usage or {}
        total_cost = getattr(result_msg, 'total_cost_usd', None)

        # Extract token counts
        input_tokens = usage.get('input_tokens', 0)
        output_tokens = usage.get('output_tokens', 0)
        cache_read = usage.get('cache_read_input_tokens', 0)
        cache_creation = usage.get('cache_creation_input_tokens', 0)

        # Calculate cache hit rate
        total_input = input_tokens + cache_read + cache_creation
        cache_rate = (cache_read / total_input * 100) if total_input > 0 else 0.0

        return BaseAgent.format_usage_line(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            prompt_tokens=total_input,
            total_cost_usd=total_cost,
        )

    def _make_event_logger(self, log_path: str | Path) -> Callable[[Any], None]:
        """Create a callback to log Claude SDK messages to JSONL.

        Args:
            log_path: Path to write JSONL log

        Returns:
            Callback function for logging events
        """
        def serializer(message: Any) -> Optional[dict[str, Any]]:
            """Serialize Claude SDK messages to JSON-compatible dict."""
            try:
                if isinstance(message, AssistantMessage):
                    # Log assistant messages with content blocks
                    blocks = []
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            blocks.append({
                                "type": "text",
                                "text": block.text
                            })
                        elif isinstance(block, ToolUseBlock):
                            blocks.append({
                                "type": "tool_use",
                                "id": block.id,
                                "name": block.name,
                                "input": block.input
                            })
                        elif isinstance(block, ToolResultBlock):
                            blocks.append({
                                "type": "tool_result",
                                "tool_use_id": block.tool_use_id,
                                "content": block.content,
                                "is_error": block.is_error
                            })

                    return {
                        "type": "assistant_message",
                        "model": getattr(message, 'model', None),
                        "content": blocks
                    }

                elif isinstance(message, ResultMessage):
                    # Log final result with metadata
                    return {
                        "type": "result",
                        "subtype": message.subtype,
                        "duration_ms": message.duration_ms,
                        "is_error": message.is_error,
                        "num_turns": message.num_turns,
                        "session_id": message.session_id,
                        "total_cost_usd": message.total_cost_usd,
                        "usage": message.usage,
                        "result": message.result
                    }

                elif hasattr(message, 'subtype') and hasattr(message, 'data'):
                    # Log system messages
                    return {
                        "type": "system_message",
                        "subtype": message.subtype,
                        "data": message.data
                    }

            except Exception:
                # Skip messages that can't be serialized
                pass

            return None

        return BaseAgent.make_jsonl_logger(log_path, serializer=serializer)

    def run(
        self,
        prompt: str,
        *,
        workspace: str | Path | None = None,
        callbacks: Optional[Iterable[Callable[[Any], None]]] = None,
        log_jsonl_path: Optional[str | Path] = None,
        answer_path: Optional[str | Path] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run Claude agent on the given prompt.

        Args:
            prompt: User instruction or task prompt
            workspace: Working directory/repo path (used as cwd)
            callbacks: Optional iterable of event callbacks (called for each message)
            log_jsonl_path: If provided, log all events to this JSONL file
            answer_path: If provided, write final answer to this file
            **kwargs: Additional Claude-specific options (merged with config)

        Returns:
            dict with keys: answer, usage, conversation_id, log_path, answer_path, metadata

        Raises:
            CLINotFoundError: If Claude Code CLI is not installed
            ProcessError: If the Claude process fails
            ValueError: If API key is missing
        """
        # Run the async implementation
        return asyncio.run(self._run_async(
            prompt=prompt,
            workspace=workspace,
            callbacks=callbacks,
            log_jsonl_path=log_jsonl_path,
            answer_path=answer_path,
            **kwargs
        ))

    async def _run_async(
        self,
        prompt: str,
        *,
        workspace: str | Path | None = None,
        callbacks: Optional[Iterable[Callable[[Any], None]]] = None,
        log_jsonl_path: Optional[str | Path] = None,
        answer_path: Optional[str | Path] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Async implementation of run().

        This is the core implementation that uses Claude SDK's async API.
        Returns a dict compatible with OpenHandsAgent output format.
        """
        # Build options
        options = self._build_options(workspace)

        # Merge any additional kwargs into options
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(options, key):
                    setattr(options, key, value)

        # Prepare callbacks
        cb_list = list(callbacks) if callbacks else []
        if log_jsonl_path:
            cb_list.append(self._make_event_logger(log_jsonl_path))

        # Collect all messages for answer extraction
        all_messages = []
        result_message = None

        try:
            # Use ClaudeSDKClient for the conversation
            async with ClaudeSDKClient(options=options) as client:
                # Send the prompt
                await client.query(prompt)

                # Receive all messages until completion
                async for message in client.receive_response():
                    all_messages.append(message)

                    # Call custom callbacks
                    for callback in cb_list:
                        try:
                            callback(message)
                        except Exception:
                            pass  # Best-effort callbacks

                    # Capture the final result
                    if isinstance(message, ResultMessage):
                        result_message = message

        except CLINotFoundError as e:
            return {
                "answer": f"Error: Claude Code CLI not found. Install with: npm install -g @anthropic-ai/claude-code\n{str(e)}",
                "usage": "",
                "answer_path": None,
                "log_path": str(log_jsonl_path) if log_jsonl_path else None,
                "conversation_id": None,
                "metadata": {"error": str(e), "error_type": "CLINotFoundError"}
            }
        except ProcessError as e:
            error_msg = f"Error: Claude process failed (exit code {e.exit_code})\n"
            if e.stderr:
                error_msg += f"STDERR: {e.stderr}\n"
            error_msg += str(e)

            return {
                "answer": error_msg,
                "usage": "",
                "answer_path": None,
                "log_path": str(log_jsonl_path) if log_jsonl_path else None,
                "conversation_id": None,
                "metadata": {"error": str(e), "error_type": "ProcessError", "exit_code": e.exit_code}
            }
        except Exception as e:
            return {
                "answer": f"Error: {type(e).__name__}: {str(e)}",
                "usage": "",
                "answer_path": None,
                "log_path": str(log_jsonl_path) if log_jsonl_path else None,
                "conversation_id": None,
                "metadata": {"error": str(e), "error_type": type(e).__name__}
            }

        # Extract final answer
        answer = self._extract_final_answer(all_messages)
        if not answer and result_message and hasattr(result_message, 'result'):
            answer = result_message.result or ""

        # Format usage
        usage = ""
        if result_message:
            usage = self._format_usage_from_result(result_message)

        # Save answer to file if requested
        saved_answer_path = None
        if answer_path and answer:
            saved_answer_path = self.save_text(answer_path, answer)

        # Extract session ID from result
        session_id = None
        if result_message and hasattr(result_message, 'session_id'):
            session_id = result_message.session_id

        # Build metadata
        metadata = {}
        if result_message:
            metadata.update({
                "duration_ms": getattr(result_message, 'duration_ms', None),
                "num_turns": getattr(result_message, 'num_turns', None),
                "is_error": getattr(result_message, 'is_error', False),
            })

        # Return as dict for compatibility with main.py
        # (OpenHandsAgent also returns dict)
        return {
            "answer": answer,
            "usage": usage,
            "answer_path": saved_answer_path,
            "log_path": str(log_jsonl_path) if log_jsonl_path else None,
            "conversation_id": session_id,
            "metadata": metadata
        }


# Convenience function for quick usage
def run_claude_query(
    prompt: str,
    *,
    workspace: str | Path | None = None,
    model: str = "claude-3-5-sonnet-20241022",
    allowed_tools: Optional[list[str]] = None,
    permission_mode: str = "acceptEdits",
    api_key: Optional[str] = None,
    log_path: Optional[str | Path] = None,
    answer_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Quick convenience function to run a Claude query.

    Args:
        prompt: Task prompt for Claude
        workspace: Working directory
        model: Claude model name
        allowed_tools: List of allowed tools (defaults to Read, Write, Edit, Bash, Grep, Glob)
        permission_mode: Permission mode (default "acceptEdits")
        api_key: API key (reads from env if None)
        log_path: Optional JSONL log path
        answer_path: Optional answer output path

    Returns:
        dict with keys: answer, usage, conversation_id, log_path, answer_path, metadata

    Example:
        result = run_claude_query(
            "Analyze this code for security issues",
            workspace="/path/to/repo",
            model="claude-3-5-sonnet-20241022",
            answer_path="./output/answer.txt"
        )
        print(result["answer"])
    """
    tools = allowed_tools or ["Read", "Write", "Edit", "Bash", "Grep", "Glob"]

    agent = ClaudeAgent(
        model_name=model,
        api_key=api_key,
        permission_mode=permission_mode,
        allowed_tools=tools,
    )

    return agent.run(
        prompt=prompt,
        workspace=workspace,
        log_jsonl_path=log_path,
        answer_path=answer_path,
    )
