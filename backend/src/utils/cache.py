import json
import os
from typing import Any

CACHE_DIR = "/tmp/repo_cache"
# CACHE_TTL_SECONDS = 6 * 60 * 60  # 6 hours
CACHE_TTL_SECONDS = 0
CACHE_MAX_FILES = 100

def _enforce_lru_cache_limit():
    files = [
        (f, os.path.getatime(os.path.join(CACHE_DIR, f)))
        for f in os.listdir(CACHE_DIR)
        if f.endswith(".json")
    ]
    if len(files) > CACHE_MAX_FILES:
        files.sort(key=lambda x: x[1])  # Oldest access time first
        for f, _ in files[:len(files) - CACHE_MAX_FILES]:
            os.remove(os.path.join(CACHE_DIR, f))

def get_cache_path(owner: str, repo: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{owner}_{repo}.json")

def load_repo_cache(owner: str, repo: str) -> dict[str, Any] | None:
    path = get_cache_path(owner, repo)
    if os.path.exists(path):
        os.utime(path, None)  # Update access time for LRU
        with open(path, "r") as f:
            data = json.load(f)
            cached_at = data.get("cached_at", 0)
            import time
            if time.time() - cached_at < CACHE_TTL_SECONDS:
                return data
    return None

def save_repo_cache(owner: str, repo: str, summary: Any, tree: Any, content: Any) -> None:
    path = get_cache_path(owner, repo)
    import time
    with open(path, "w") as f:
        json.dump({"summary": summary, "tree": tree, "content": content, "cached_at": time.time()}, f)
    _enforce_lru_cache_limit()
