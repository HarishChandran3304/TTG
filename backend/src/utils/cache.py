import os
import json
from typing import Any
import time

CACHE_DIR = "/tmp/repo_cache"
CACHE_TTL_SECONDS = 6 * 60 * 60  # 6 hours

def get_cache_path(owner: str, repo: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{owner}_{repo}.json")

def load_repo_cache(owner: str, repo: str) -> dict[str, Any] | None:
    path = get_cache_path(owner, repo)
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
            cached_at = data.get("cached_at", 0)
            if time.time() - cached_at < CACHE_TTL_SECONDS:
                return data
    return None

def save_repo_cache(owner: str, repo: str, summary: Any, tree: Any, content: Any) -> None:
    path = get_cache_path(owner, repo)
    with open(path, "w") as f:
        json.dump({"summary": summary, "tree": tree, "content": content, "cached_at": time.time()}, f)
