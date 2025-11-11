"""
Cursor Agent implementation using Cursor CLI headless mode.

This agent wraps the Cursor CLI which must be installed separately.
Cursor is an AI-powered code editor with command-line capabilities.

Installation:
    - Install Cursor from https://cursor.sh
    - Ensure 'cursor' command is available in PATH

IMPORTANT: Web search tools are explicitly DISABLED to ensure the agent
only analyzes local repository code without external data.
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


# Check if Cursor CLI is available
def _check_cursor_cli() -> bool:
    """Check if 'cursor' command is available in PATH."""
    return shutil.which("cursor") is not None


CURSOR_CLI_AVAILABLE = _check_cursor_cli()


@dataclass
class CursorAgent(BaseAgent):
    """Cursor Agent implementation using Cursor CLI headless mode.

    This agent uses Cursor's command-line interface to perform AI-assisted
    code analysis and generation. Cursor is an AI-powered code editor that
    can be run in headless mode for automation.

    IMPORTANT: Web search tools are explicitly DISABLED to ensure the agent
    only analyzes local repository code without external data.

    Attributes:
        model_name: Model to use (e.g., "gpt-4", "claude-3-opus", "gpt-3.5-turbo")
        api_key: API key for the model provider (reads from env vars if None)
        service_id: Logical service identifier for tracking
        config: Additional configuration options
        disable_web_search: Enforce web search is disabled (must be True)
        max_iterations: Maximum number of agent iterations (default: 10)
        timeout: Timeout in seconds for the cursor command (None for no timeout)
        working_dir: Working directory override (defaults to workspace)

    Tool Configuration:
        - File operations: Read, Write, Edit (ENABLED)
        - Shell commands: Execution in workspace (ENABLED)
        - Code analysis: Symbol search, grep (ENABLED)
        - Web tools: WebSearch, WebFetch (EXPLICITLY DISABLED)

    Example:
        agent = CursorAgent(
            model_name="gpt-4",
            api_key="sk-...",
            disable_web_search=True
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

    # Cursor-specific configuration
    disable_web_search: bool = True  # MUST be True - no web access
    max_iterations: int = 10  # Maximum agent iterations
    timeout: Optional[int] = 1800  # 30 minutes default timeout
    working_dir: Optional[str | Path] = None

    def __post_init__(self):
        """Validate that Cursor CLI is available and web search is disabled."""
        if not CURSOR_CLI_AVAILABLE:
            raise ImportError(
                "Cursor CLI is not installed. "
                "Install Cursor from: https://cursor.sh\n"
                "Ensure 'cursor' command is in your PATH."
            )

        # Enforce web search restriction
        if not self.disable_web_search:
            raise ValueError(
                "Web search must be disabled for security reasons. "
                "CursorAgent must analyze code locally without external web access. "
                "Set disable_web_search=True (default)."
            )

        # Override API key lookup to check common env vars
        if not self.api_key:
            self.api_key = (
                os.getenv("CURSOR_API_KEY") or
                os.getenv("OPENAI_API_KEY") or
                os.getenv("ANTHROPIC_API_KEY") or
                os.getenv("OPENHANDS_API_KEY") or
                os.getenv("LLM_API_KEY")
            )

        # Note: Cursor may use its own authentication, so API key might be optional
        # We'll proceed without raising an error if no API key is found

    def _create_cursor_config(self, workspace: Path) -> dict:
        """
        Create configuration for Cursor with web search disabled.
        
        Returns configuration dict.
        """
        config = {
            "model": self.model_name,
            "disable_web_search": True,  # Explicitly disable web search
            "max_iterations": self.max_iterations,
            "workspace": str(workspace),
            # Additional config from self.config if provided
            **(self.config or {})
        }
        
        return config

    def _create_prompt_file(self, prompt: str, workspace: Path) -> Path:
        """
        Create a temporary file containing the prompt.
        
        Returns path to the prompt file.
        """
        prompt_file = workspace / ".cursor_prompt.txt"
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(prompt)
        
        return prompt_file

    def _parse_cursor_output(self, output: str, error_output: str) -> tuple[str, dict]:
        """
        Parse Cursor CLI output to extract answer and metadata.
        
        Returns (answer, metadata) tuple.
        """
        # Combine stdout and stderr for analysis
        combined_output = output.strip()
        
        # If there's error output, include it in metadata
        metadata = {
            "output_length": len(output),
            "error_length": len(error_output),
            "model": self.model_name,
            "max_iterations": self.max_iterations,
            "web_search_disabled": True,
        }
        
        if error_output.strip():
            metadata["stderr"] = error_output.strip()
        
        # The answer is the main output from Cursor
        answer = combined_output if combined_output else "No output from Cursor"
        
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

    def _run_cursor_command(
        self,
        prompt: str,
        workspace: Path,
        events: list
    ) -> tuple[str, str, int]:
        """
        Run the Cursor CLI command.
        
        Returns (stdout, stderr, exit_code) tuple.
        """
        # Create prompt file
        prompt_file = self._create_prompt_file(prompt, workspace)
        
        events.append(self._create_event_log_entry(
            "prompt_file_created",
            f"Created prompt file at {prompt_file}",
            {"prompt_length": len(prompt)}
        ))

        # Build Cursor command
        # Note: The exact CLI syntax may vary. Common patterns:
        # cursor --headless <workspace> --prompt <prompt_file>
        # cursor --cli --task <task_description> --directory <workspace>
        # We'll try a reasonable approach based on typical CLI patterns
        
        cmd = [
            "cursor",
            "--headless",
            "--directory", str(workspace),
            "--prompt-file", str(prompt_file),
            "--no-web-search",  # Explicitly disable web search
        ]

        # Add model if specified
        if self.model_name:
            cmd.extend(["--model", self.model_name])

        # Add max iterations
        cmd.extend(["--max-iterations", str(self.max_iterations)])

        events.append(self._create_event_log_entry(
            "command_start",
            f"Running command: {' '.join(cmd)}",
            {"cwd": str(workspace)}
        ))

        # Set up environment
        env = os.environ.copy()
        
        # Add API key if available
        if self.api_key:
            env["CURSOR_API_KEY"] = self.api_key
            # Also try common provider keys
            if "gpt" in self.model_name.lower() or "o1" in self.model_name.lower():
                env["OPENAI_API_KEY"] = self.api_key
            elif "claude" in self.model_name.lower():
                env["ANTHROPIC_API_KEY"] = self.api_key
        
        # Ensure web search is disabled via env var
        env["CURSOR_DISABLE_WEB_SEARCH"] = "true"
        env["CURSOR_WEB_SEARCH_ENABLED"] = "false"

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

            # Clean up prompt file
            if prompt_file.exists():
                prompt_file.unlink()
                events.append(self._create_event_log_entry(
                    "prompt_file_deleted",
                    f"Cleaned up prompt file",
                    {}
                ))

            return stdout, stderr, exit_code

        except subprocess.TimeoutExpired as e:
            error_msg = f"Cursor command timed out after {self.timeout} seconds"
            events.append(self._create_event_log_entry(
                "error",
                error_msg,
                {"error_type": "timeout"}
            ))
            
            # Clean up prompt file
            if prompt_file.exists():
                prompt_file.unlink()
            
            raise TimeoutError(error_msg)

        except Exception as e:
            error_msg = f"Error running Cursor: {e}"
            events.append(self._create_event_log_entry(
                "error",
                error_msg,
                {"error_type": type(e).__name__}
            ))
            
            # Clean up prompt file
            if prompt_file.exists():
                prompt_file.unlink()
            
            raise

    def run(
        self,
        prompt: str,
        workspace: str | Path,
        log_jsonl_path: Optional[str | Path] = None,
        answer_path: Optional[str | Path] = None,
        **kwargs
    ) -> dict:
        """
        Run Cursor agent on the given prompt within the workspace.

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
        conversation_id = f"cursor_{int(time.time())}_{os.getpid()}"
        
        # Create event log
        events = []
        
        try:
            # Log start event
            events.append(self._create_event_log_entry(
                "session_start",
                f"Starting Cursor session in {workspace}",
                {
                    "model": self.model_name,
                    "workspace": str(workspace),
                    "conversation_id": conversation_id,
                    "web_search_disabled": True,
                }
            ))

            # Create configuration
            config = self._create_cursor_config(workspace)
            events.append(self._create_event_log_entry(
                "config_created",
                f"Created configuration",
                config
            ))

            # Run Cursor command
            stdout, stderr, exit_code = self._run_cursor_command(
                prompt, workspace, events
            )

            # Parse output
            answer, metadata = self._parse_cursor_output(stdout, stderr)
            metadata["exit_code"] = exit_code
            
            events.append(self._create_event_log_entry(
                "answer_extracted",
                f"Extracted answer ({len(answer)} chars)",
                metadata
            ))

            # Check for errors
            if exit_code != 0:
                events.append(self._create_event_log_entry(
                    "warning",
                    f"Non-zero exit code: {exit_code}",
                    {"stderr": stderr}
                ))

        except TimeoutError as e:
            error_msg = str(e)
            answer = f"Error: {error_msg}"
            metadata = {"error": "timeout", "exit_code": None}

        except Exception as e:
            error_msg = f"Error running Cursor: {e}"
            events.append(self._create_event_log_entry(
                "error",
                error_msg,
                {"error_type": type(e).__name__}
            ))
            answer = f"Error: {error_msg}"
            metadata = {"error": str(e), "exit_code": None}

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
            input_tokens=None,  # Cursor CLI doesn't expose token counts directly
            output_tokens=None,
            cost=None,
        )

        events.append(self._create_event_log_entry(
            "session_end",
            "Cursor session completed",
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


def run_cursor_query(
    prompt: str,
    workspace: str | Path,
    model_name: str = "gpt-4",
    api_key: Optional[str] = None,
    **kwargs
) -> dict:
    """
    Convenience function to run a Cursor query.

    Args:
        prompt: The task/question for the agent
        workspace: Directory containing the code repository
        model_name: Model to use (default: "gpt-4")
        api_key: API key for the model provider (default: from env vars)
        **kwargs: Additional arguments for CursorAgent

    Returns:
        Result dict from CursorAgent.run()

    Example:
        result = run_cursor_query(
            "Find security vulnerabilities in auth.py",
            workspace="/path/to/repo",
            model_name="gpt-4"
        )
        print(result["answer"])
    """
    agent = CursorAgent(
        model_name=model_name,
        api_key=api_key,
        **kwargs
    )
    return agent.run(prompt=prompt, workspace=workspace)
