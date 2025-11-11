import subprocess
import json

def exec_command(cmd, cwd="."):
    """Executes a shell command and returns the output."""
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    out, err = process.communicate()
    return out, err, process.returncode

def clone_repo(repo_url: str, dest_dir: str) -> None:
    """Clones a git repository to the specified destination directory."""
    cmd = f"git clone {repo_url}"
    out, err, returncode = exec_command(cmd, cwd=dest_dir)
    if returncode != 0:
        raise Exception(f"Error cloning repository {cmd}: {err}")

def pull_repo(repo_dir: str) -> None:
    """Pulls the latest changes in the specified git repository directory."""
    cmd = "git pull"
    out, err, returncode = exec_command(cmd, cwd=repo_dir)
    if returncode != 0:
        raise Exception(f"Error pulling repository {cmd}: {err}")

def checkout_to_commit(repo_dir: str, commit_hash: str) -> None:
    """Checks out the specified commit in the given git repository directory."""
    cmd = f"git checkout {commit_hash}"
    out, err, returncode = exec_command(cmd, cwd=repo_dir)
    if returncode != 0:
        raise Exception(f"Error checking out commit {commit_hash}: {err}")

def load_jsonl(file_path: str) -> list:
    """Loads a JSONL file and returns a list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data