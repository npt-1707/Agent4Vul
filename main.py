#!/usr/bin/env python3
"""
Main entry point for vulnerability detection using various agent platforms.

This script orchestrates the entire pipeline:
  1. Parses CLI arguments (agent name, model, dataset, indices, output dir)
  2. Loads the specified dataset and clones/pulls repos
  3. Instantiates the requested agent (OpenHands, Claude, Cursor, etc.)
  4. For each dataset item:
     - Extracts vulnerable functions and commit hash
     - Builds a prompt from the template
     - Runs the agent on the cloned repo workspace
     - Saves the answer, usage metrics, and event log

Usage:
    python main.py --agent openhands --model openhands/claude-sonnet-4-5-20250929 \\
                   --dataset ReposVul --start 0 --end 10 --output ./outputs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

# Allow imports from repo_level_vuldetection package
_here = Path(__file__).resolve()
_repo_root = _here.parent  # repo_level_vuldetection/
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from data_processing.load_data import (
    load_ReposVul_dataset,
    get_ReposVul_dataitem_vuln_func,
    get_ReposVul_dataitem_commit_hash,
)
from data_processing.util import checkout_to_commit
from agents.openhands_agent import OpenHandsAgent
from agents.prompts import instructions

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file if present


def convert_jsonl_to_json(jsonl_path: str, json_path: str):
    """Convert a JSONL file to a JSON file containing a list of records."""
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            records.append(record)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

def build_prompt(repo_path: str, commit_hash: str, function_source: str) -> str:
    """Build the vulnerability detection prompt from the template."""
    return "".join(instructions).format(path_to_repo=repo_path, commit_hash=commit_hash, function_source_code=function_source)


def get_agent(agent_name: str, model_name: str, **kwargs):
    """Factory function to instantiate the requested agent."""
    agent_name_lower = agent_name.lower()
    
    if agent_name_lower == "openhands":
        return OpenHandsAgent(
            llm_model_name=model_name,
            **kwargs
        )
    elif agent_name_lower == "claude":
        from agents.claude_agent import ClaudeAgent
        return ClaudeAgent(
            model_name=model_name,
            **kwargs
        )
    elif agent_name_lower == "codex":
        from agents.codex_agent import CodexAgent
        return CodexAgent(
            model_name=model_name,
            **kwargs
        )
    elif agent_name_lower == "cursor":
        from agents.cursor_agent import CursorAgent
        return CursorAgent(
            model_name=model_name,
            **kwargs
        )
    else:
        raise ValueError(
            f"Unknown agent: {agent_name}. "
            f"Supported agents: openhands, claude, codex, cursor."
        )


def load_dataset(dataset_name: str, start_idx: int, end_idx: int) -> list:
    """Load the specified dataset and return the list of items."""
    dataset_name_lower = dataset_name.lower()
    
    if dataset_name_lower == "reposvul":
        return load_ReposVul_dataset(start_idx=start_idx, end_idx=end_idx)
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Supported datasets: ReposVul."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run vulnerability detection agents on repository datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Core arguments
    parser.add_argument(
        "--agent",
        type=str,
        default="openhands",
        help="Agent platform to use (openhands, claude, cursor, etc.)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openhands/claude-sonnet-4-5-20250929",
        help="Model name/identifier for the agent's LLM",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ReposVul",
        help="Dataset to load (ReposVul, etc.)",
    )
    
    # Dataset range
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index in the dataset (inclusive)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=10,
        help="End index in the dataset (exclusive)",
    )
    
    # Output directory
    parser.add_argument(
        "--output",
        type=str,
        default="./output",
        help="Directory to save agent outputs (answers, logs, usage)",
    )
    
    # # API key (optional, falls back to env vars)
    # parser.add_argument(
    #     "--api-key",
    #     type=str,
    #     default=None,
    #     help="API key for the agent (if not set via OPENHANDS_API_KEY env var)",
    # )
    
    # # Advanced agent options
    # parser.add_argument(
    #     "--service-id",
    #     type=str,
    #     default="vuln-detection-agent",
    #     help="Logical service ID for metrics/tracing",
    # )
    # parser.add_argument(
    #     "--base-url",
    #     type=str,
    #     default=None,
    #     help="Custom base URL for LLM API (proxy or self-hosted)",
    # )
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print(f"Agent4Vul - Vulnerability Detection Pipeline")
    print("=" * 80)
    print(f"Agent:    {args.agent}")
    print(f"Model:    {args.model}")
    print(f"Dataset:  {args.dataset}")
    print(f"Range:    [{args.start}, {args.end})")
    print(f"Output:   {output_dir.resolve()}")
    print("=" * 80)
    print()
    
    # Load dataset
    print(f"Loading {args.dataset} dataset (items {args.start} to {args.end})...")
    dataset = load_dataset(args.dataset, args.start, args.end)
    items_to_process = dataset[args.start:args.end]
    print(f"✓ Loaded {len(items_to_process)} items\n")
    
    # Instantiate agent
    print(f"Initializing {args.agent} agent with model {args.model}...")
    agent_kwargs = {
        # "service_id": args.service_id,
    }
    API_KEY = os.getenv("OPENHANDS_API_KEY") or os.getenv("API_KEY")
    if API_KEY:
        agent_kwargs["api_key"] = API_KEY
    # if args.base_url:
    #     agent_kwargs["base_url"] = args.base_url
    
    agent = get_agent(args.agent, args.model, **agent_kwargs)
    print(f"✓ Agent initialized\n")
    
    # Process each dataset item
    for idx, item in enumerate(items_to_process, start=args.start):
        project = item.get("project", f"unknown_{idx}")
        print(f"[{idx}/{args.end - 1}] Processing: {project}")
        
        # Extract data
        vuln_funcs = get_ReposVul_dataitem_vuln_func(item)
        commit_hash = get_ReposVul_dataitem_commit_hash(item)
        repo_path = item.get("local_repo_path")
        
        if not repo_path or not Path(repo_path).exists():
            print(f"  ⚠ Skipping: repo path not found or not cloned: {repo_path}")
            continue
        
        if not vuln_funcs:
            print(f"  ⚠ Skipping: no vulnerable functions found in item")
            continue
        
        # Process each vulnerable function
        for func_idx, func in enumerate(vuln_funcs):
            func_code = func.get("function", "")
            
            if not func_code.strip():
                print(f"    ⚠ Skipping function: empty code")
                continue
            
            # print(f"  → Analyzing function: {func_name}")
            
            # Build prompt
            prompt = build_prompt(repo_path, commit_hash, func_code)

            # Checkout to the commit hash
            checkout_to_commit(repo_path, commit_hash)
            
            # Output paths
            safe_project = project.replace("/", "_")
            base_filename = f"{safe_project}_{commit_hash[:8]}"
            
            answer_file = output_dir / f"answer_{base_filename}.txt"
            jsonl_log_file = output_dir / f"log_{base_filename}.jsonl"
            json_log_file = output_dir / f"log_{base_filename}.json"
            usage_file = output_dir / f"usage_{base_filename}.txt"
            
            # Run agent
            try:
                result = agent.run(
                    prompt=prompt,
                    workspace=repo_path,
                    log_jsonl_path=str(jsonl_log_file),
                    answer_path=str(answer_file),
                )
                
                # Save usage separately
                if result.get("usage"):
                    usage_file.write_text(result["usage"], encoding="utf-8")
                convert_jsonl_to_json(str(jsonl_log_file), str(json_log_file))
                print(f"    ✓ Answer:  {answer_file.name}")
                print(f"    ✓ Log:     {json_log_file.name}")
                print(f"    ✓ Usage:   {usage_file.name}")
                if result.get("usage"):
                    print(f"    ℹ {result['usage']}")
                
            except Exception as e:
                print(f"    ✗ Error running agent: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print()
    
    print("=" * 80)
    print("✓ Pipeline completed")
    print(f"Results saved to: {output_dir.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
