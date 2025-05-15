import os
import shutil
from pathlib import Path

from git import Repo
from dotenv import load_dotenv
from gitingest import ingest_async  # type: ignore
import aiohttp

from src.utils.llm import generate_response
from src.utils.prompt import generate_prompt

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")


async def check_repo_exists(owner: str, name: str) -> bool:
    """Check if a repository exists and is accessible."""
    api_url = f"https://api.github.com/repos/{owner}/{name}"
    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    connector = aiohttp.TCPConnector(ssl=False)

    async with aiohttp.ClientSession(connector=connector) as session:
        try:
            response = await session.get(api_url, headers=headers)
            data = await response.json()
            if response.status == 200:
                return True
            elif response.status == 404:
                return False
            elif "message" in data and "API rate limit exceeded" in data["message"]:
                raise ValueError("error:rate_limit")
            else:
                return False
        except Exception as e:
            print(f"Error checking repo: {e}")
            return False


async def ingest_repo(owner: str, name: str) -> tuple[str, str, str]:
    """
    Converts a github repository into LLM-friendly format.

    Args:
            owner: owner of repository
            name: name of repository
    Returns:
            A tuple containing the summary, the folder structure, and the content of the files in LLM-friendly format.
    """
    # Check if repository exists and is accessible
    # if not await check_repo_exists(owner, name):
    #     raise ValueError("error:repo_not_found")

    try:
        repo_path = f"/tmp/repo/{owner}-{name}"
        summary, tree, content = await ingest_async(
            repo_path, exclude_patterns={
                "*/test",
                "*gradle*",
                "*libs*",
                "*.xlsx",
                "*/manifests",
                "Makefile",
                ".github",
                ".java"
            }
        )

        # Check if token count exceeds limit
        # if "Estimated tokens: " in summary:
        #     tokens_str = summary.split("Estimated tokens: ")[-1].strip()
        #     if tokens_str.endswith("M"):
        #         raise ValueError("error:repo_too_large")
        #     elif tokens_str.endswith("K"):
        #         tokens = float(tokens_str[:-1])
        #         if tokens > 750:
        #             raise ValueError("error:repo_too_large")

        return summary, tree, content
    except Exception as e:
        if "Repository not found" in str(e) or "Not Found" in str(e):
            raise ValueError("error:repo_not_found")
        if "Bad credentials" in str(e) or "API rate limit exceeded" in str(e):
            raise ValueError("error:repo_private")
        raise


def pull_repo(owner, name):
    try:
        repo_url = f"https://{GITHUB_TOKEN}@github.com/{owner}/{name}.git"
        repo_path = f"/tmp/repo/{owner}_{name}"
        if os.path.exists(repo_path) and os.path.isdir(os.path.join(repo_path, '.git')):
            print(f"repository {repo_path} already exists.")
            repo = Repo(repo_path)
            origin = repo.remotes.origin
            origin.pull()
            print(f"pull repository({owner}/{name}) success. (HEAD: {repo.head.commit.hexsha})")
            return True

        print(f"clone repository...")
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path)

        os.makedirs(repo_path, exist_ok=True)

        repo = Repo.clone_from(repo_url, repo_path)
        print(f"clone repository({owner}/{name}) success. (HEAD: {repo.head.commit.hexsha})")
        return True
    except Exception as e:
        print(f"repository pull failed. {e}")
        return False


if __name__ == "__main__":
    import asyncio


    async def main():
        # exists = await check_repo_exists("fastlane-dev", "yeoshin-backend-v2")
        # print(exists)
        # pull_repo("fastlane-dev", "yeoshin-backend-v2")
        summary, tree, content = await ingest_repo("fastlane-dev", "yeoshin-backend-v2")
        print(summary)
        print(tree)
        print(content)
        prompt = await generate_prompt("아무말", [], tree, "아무말")
        response = await generate_response(prompt)

        print(response)


    asyncio.run(main())
