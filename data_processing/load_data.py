from .util import clone_repo, pull_repo, load_jsonl
import os
from pathlib import Path
from tqdm import tqdm

# Resolve paths relative to the repository root, regardless of CWD
_here = Path(__file__).resolve()
_repo_root = _here.parent.parent  # repo_level_vuldetection/
clone_path = _repo_root / "dataset" / "cloned_repos"
clone_path.mkdir(parents=True, exist_ok=True)

def load_ReposVul_dataset(start_idx: int=0, end_idx: int=100) -> list:
    """Loads the ReposVul dataset from the specified path and clones repositories."""
    print("Preparing ReposVul dataset...")
    ReposVul_dataset_jsonl_path = _repo_root / "dataset" / "ReposVul.jsonl"
    data = load_jsonl(str(ReposVul_dataset_jsonl_path))
    for item in tqdm(data[start_idx:end_idx]):
        project = item["project"]
        repo_owner, repo_name = project.split("/")
        clone_url = f"https://github.com/{repo_owner}/{repo_name}.git"
        dest_dir = clone_path / repo_owner / repo_name
        if not dest_dir.exists():
            base_dir = dest_dir.parent
            base_dir.mkdir(parents=True, exist_ok=True)
            clone_repo(clone_url, str(base_dir))
        else:
            pull_repo(str(dest_dir))
        # Store the absolute path to the cloned repo
        item["local_repo_path"] = str(dest_dir)
    return data

def get_ReposVul_dataitem_vuln_func(item: dict) -> list:
    """Extracts vulnerable functions from a ReposVul dataset item."""
    vuln_funcs = []
    for file in item["details"]:
        for func in file["function_before"]:
            if func["target"] == 1:
                vuln_funcs.append(func)
        return vuln_funcs

def get_ReposVul_dataitem_commit_hash(item: dict) -> str:
    """Extracts the commit hash from a ReposVul dataset item."""
    return item["url"].split("/")[-1]