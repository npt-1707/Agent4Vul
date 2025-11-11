"""
Codex Agent implementation using OpenAI Codex CLI.

This agent wraps the Codex CLI (https://github.com/openai/codex) which must be
installed separately via:
    npm install -g @openai/codex

The agent runs Codex in non-interactive "exec" mode and captures all output.

IMPORTANT: Web search tools are explicitly DISABLED via configuration to ensure
the agent only analyzes local repository code without external data.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from agents.base_agent import BaseAgent


# Check if Codex CLI is available
def _check_codex_cli() -> bool:
    """Check if 'codex' command is available in PATH."""
    return shutil.which("codex") is not None


CODEX_CLI_AVAILABLE = _check_codex_cli()


@dataclass
class CodexAgent(BaseAgent):
    """Codex Agent implementation using OpenAI Codex CLI.

    This agent uses the Codex CLI from OpenAI which provides an AI coding
    agent that can read/write files, run commands, and analyze code locally.

    IMPORTANT: Web search tools are explicitly DISABLED to ensure the agent
    only analyzes local repository code without external data.

    Attributes:
        model_name: Model to use (e.g., "o4-mini", "gpt-5-codex", "claude-3-5-sonnet")
        api_key: OpenAI API key (reads from OPENAI_API_KEY if None)
        service_id: Logical service identifier for tracking
        config: Additional configuration options
        approval_policy: Approval policy for commands ("never", "untrusted", "on-failure", "on-request")
        sandbox: Sandbox mode ("read-only", "workspace-write", "danger-full-access")
        web_search_enabled: MUST be False - web search is disabled for security
        max_turns: Maximum conversation turns (None for default)
        timeout: Timeout in seconds for the codex exec command (None for no timeout)

    Tool Configuration:
        - File operations: Read, Write, Edit (ENABLED)
        - Shell commands: Bash execution (ENABLED)
        - Search tools: Grep, file search (ENABLED)
        - Web tools: WebSearch, WebFetch (EXPLICITLY DISABLED)

    Example:
        agent = CodexAgent(
            model_name="o4-mini",
            api_key="sk-...",
            approval_policy="never",
            sandbox="workspace-write"
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

    # Codex-specific configuration
    approval_policy: str = "never"  # "never", "untrusted", "on-failure", "on-request"
    sandbox: str = "workspace-write"  # "read-only", "workspace-write", "danger-full-access"
    web_search_enabled: bool = False  # MUST be False - no web access
    max_turns: Optional[int] = None
    timeout: Optional[int] = 1800  # 30 minutes default timeout

    def __post_init__(self):
        """Validate that Codex CLI is available and web search is disabled."""
        if not CODEX_CLI_AVAILABLE:
            raise ImportError(
                "Codex CLI is not installed. "
                "Install it with: npm install -g @openai/codex\n"
                "See: https://github.com/openai/codex"
            )

        # Enforce web search restriction
        if self.web_search_enabled:
            raise ValueError(
                "Web search is not allowed for security reasons. "
                "CodexAgent must analyze code locally without external web access. "
                "Set web_search_enabled=False (default)."
            )

        # Override API key lookup to check OpenAI-specific env vars
        if not self.api_key:
            self.api_key = (
                os.getenv("OPENAI_API_KEY") or
                os.getenv("CODEX_API_KEY") or
                os.getenv("OPENHANDS_API_KEY") or
                os.getenv("LLM_API_KEY")
            )

        if not self.api_key:
            raise ValueError(
                "No API key found. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

    def _create_config_file(self, workspace: Path) -> Path:
        """
        Create a temporary config.json file for Codex with web search disabled.
        
        Returns path to the config file.
        """
        config = {
            "model": self.model_name,
            "provider": self._get_provider_from_model(),
            "tools": {
                "web_search": False,  # Explicitly disable web search
                "web_search_request": False,  # Alternative name for web search
            },
            # Additional config from self.config if provided
            **(self.config or {})
        }
        
        config_path = workspace / ".codex_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        
        return config_path

    def _get_provider_from_model(self) -> str:
        """Infer provider from model name."""
        model_lower = self.model_name.lower()
        if "gpt" in model_lower or "o1" in model_lower or "o3" in model_lower or "o4" in model_lower:
            return "openai"
        elif "claude" in model_lower:
            return "anthropic"
        elif "gemini" in model_lower:
            return "gemini"
        elif "deepseek" in model_lower:
            return "deepseek"
        elif "llama" in model_lower:
            return "ollama"
        else:
            return "openai"  # default

    def _parse_codex_output(self, output: str) -> tuple[str, dict]:
        """
        Parse Codex CLI output to extract answer and metadata.
        
        Returns (answer, metadata) tuple.
        """
        # Codex exec outputs the agent's response and actions
        # We'll treat the entire output as the answer
        answer = output.strip()
        
        metadata = {
            "output_length": len(output),
            "model": self.model_name,
            "sandbox": self.sandbox,
            "approval_policy": self.approval_policy,
        }
        
        return answer, metadata

    def _create_event_log_entry(
        self,
        event_type: str,
        content: str,
        metadata: Optional[dict] = None
    ) -> dict:
        """Create a standardized event log entry."""
        entry = {
            "timestamp": time.time(),
            "event_type": event_type,
            "content": content,
        }
        if metadata:
            entry["metadata"] = metadata
        return entry

    def run(
        self,
        prompt: str,
        workspace: str | Path,
        log_jsonl_path: Optional[str | Path] = None,
        answer_path: Optional[str | Path] = None,
        **kwargs
    ) -> dict:
        """
        Run Codex agent on the given prompt within the workspace.

        Args:
            prompt: The user prompt/task for the agent
            workspace: Directory containing the code repository
            log_jsonl_path: Path to save JSONL event log
            answer_path: Path to save the final answer
            **kwargs: Additional arguments (unused, for compatibility)

        Returns:
            Dict with keys:
                - answer: Final response from the agent
                - usage: Usage/cost information (if available)
                - conversation_id: Unique ID for this run
                - log_path: Path where log was saved
                - answer_path: Path where answer was saved
                - metadata: Additional metadata
        """
        workspace = Path(workspace).resolve()
        if not workspace.exists():
            return {
                "answer": f"Error: Workspace does not exist: {workspace}",
                "usage": "N/A",
                "conversation_id": None,
                "log_path": log_jsonl_path,
                "answer_path": answer_path,
                "metadata": {"error": "workspace_not_found"},
            }

        # Generate unique conversation ID
        conversation_id = f"codex_{int(time.time())}_{os.getpid()}"
        
        # Create event log
        events = []
        
        try:
            # Log start event
            events.append(self._create_event_log_entry(
                "session_start",
                f"Starting Codex session in {workspace}",
                {
                    "model": self.model_name,
                    "workspace": str(workspace),
                    "conversation_id": conversation_id,
                }
            ))

            # Create config file with web search disabled
            config_path = self._create_config_file(workspace)
            events.append(self._create_event_log_entry(
                "config_created",
                f"Created config at {config_path}",
                {"web_search_enabled": False}
            ))

            # Build Codex command
            cmd = [
                "codex",
                "exec",
                prompt,
                "--cd", str(workspace),
                "--approval-policy", self.approval_policy,
                "--sandbox", self.sandbox,
                "-o",  # Output only final response
            ]

            # Add model flag if not default
            if self.model_name:
                cmd.extend(["--model", self.model_name])

            # Add config file
            if config_path.exists():
                cmd.extend(["--config", str(config_path)])

            events.append(self._create_event_log_entry(
                "command_start",
                f"Running command: {' '.join(cmd)}",
                {"cwd": str(workspace)}
            ))

            # Set up environment with API key
            env = os.environ.copy()
            env["OPENAI_API_KEY"] = self.api_key
            
            # Ensure web search is disabled via env var (if Codex supports it)
            env["CODEX_WEB_SEARCH_ENABLED"] = "false"

            # Run Codex CLI
            try:
                result = subprocess.run(
                    cmd,
                    cwd=workspace,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                
                stdout = result.stdout
                stderr = result.stderr
                exit_code = result.returncode

                events.append(self._create_event_log_entry(
                    "command_complete",
                    f"Command completed with exit code {exit_code}",
                    {
                        "exit_code": exit_code,
                        "stdout_length": len(stdout),
                        "stderr_length": len(stderr),
                    }
                ))

                if stderr:
                    events.append(self._create_event_log_entry(
                        "stderr",
                        stderr,
                        {"exit_code": exit_code}
                    ))

                # Parse output
                answer, metadata = self._parse_codex_output(stdout)
                
                events.append(self._create_event_log_entry(
                    "answer_extracted",
                    f"Extracted answer ({len(answer)} chars)",
                    metadata
                ))

            except subprocess.TimeoutExpired:
                error_msg = f"Codex command timed out after {self.timeout} seconds"
                events.append(self._create_event_log_entry(
                    "error",
                    error_msg,
                    {"error_type": "timeout"}
                ))
                answer = f"Error: {error_msg}"
                metadata = {"error": "timeout"}

            except Exception as e:
                error_msg = f"Error running Codex: {e}"
                events.append(self._create_event_log_entry(
                    "error",
                    error_msg,
                    {"error_type": type(e).__name__}
                ))
                answer = f"Error: {error_msg}"
                metadata = {"error": str(e)}

            # Clean up config file
            if config_path.exists():
                config_path.unlink()

        except Exception as e:
            error_msg = f"Fatal error in CodexAgent: {e}"
            events.append(self._create_event_log_entry(
                "fatal_error",
                error_msg,
                {"error_type": type(e).__name__}
            ))
            answer = f"Error: {error_msg}"
            metadata = {"error": str(e)}

        # Save event log
        if log_jsonl_path:
            log_path = Path(log_jsonl_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w", encoding="utf-8") as f:
                for event in events:
                    f.write(json.dumps(event) + "\n")

        # Save answer
        if answer_path:
            ans_path = Path(answer_path)
            ans_path.parent.mkdir(parents=True, exist_ok=True)
            with open(ans_path, "w", encoding="utf-8") as f:
                f.write(answer)

        # Format usage information
        usage = self.format_usage_line(
            model=self.model_name,
            input_tokens=None,  # Codex CLI doesn't expose token counts directly
            output_tokens=None,
            cost=None,
        )

        events.append(self._create_event_log_entry(
            "session_end",
            "Codex session completed",
            {"conversation_id": conversation_id}
        ))

        return {
            "answer": answer,
            "usage": usage,
            "conversation_id": conversation_id,
            "log_path": str(log_jsonl_path) if log_jsonl_path else None,
            "answer_path": str(answer_path) if answer_path else None,
            "metadata": metadata,
        }


def run_codex_query(
    prompt: str,
    workspace: str | Path,
    model_name: str = "o4-mini",
    api_key: Optional[str] = None,
    **kwargs
) -> dict:
    """
    Convenience function to run a Codex query.

    Args:
        prompt: The task/question for the agent
        workspace: Directory containing the code repository
        model_name: Model to use (default: "o4-mini")
        api_key: OpenAI API key (default: from OPENAI_API_KEY env var)
        **kwargs: Additional arguments for CodexAgent

    Returns:
        Result dict from CodexAgent.run()

    Example:
        result = run_codex_query(
            "Find security vulnerabilities in auth.py",
            workspace="/path/to/repo",
            model_name="o4-mini"
        )
        print(result["answer"])
    """
    agent = CodexAgent(
        model_name=model_name,
        api_key=api_key,
        **kwargs
    )
    return agent.run(prompt=prompt, workspace=workspace)
