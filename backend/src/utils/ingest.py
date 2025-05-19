import inspect
import locale
import os
import platform
import shutil
import tomllib
import warnings
from fnmatch import fnmatch
from pathlib import Path
from typing import Tuple, List, Set, Optional, Union

import aiohttp
from dotenv import load_dotenv
from git import Repo
from gitingest import parse_query, clone_repo
from gitingest.config import MAX_TOTAL_SIZE_BYTES, MAX_FILES, MAX_DIRECTORY_DEPTH, TMP_BASE_PATH
from gitingest.filesystem_schema import FileSystemNode, FileSystemNodeType, FileSystemStats
from gitingest.output_formatters import format_single_file, format_directory
from gitingest.query_parsing import ParsedQuery
from gitingest.utils.path_utils import _is_safe_symlink

from src.utils.llm import initialize_chat
from src.utils.prompt import generate_prompt

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
FASTLANE_DEV = "fastlane-dev"


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


async def ingest_repo(owner: str, name: str, sub_name: str) -> tuple[str, str, str]:
    """
    Converts a github repository into LLM-friendly format.

    Args:
            owner: owner of repository
            name: name of repository
            sub_name: sub category of repository
    Returns:
            A tuple containing the summary, the folder structure, and the content of the files in LLM-friendly format.
    """
    try:
        if not await pull_repo(owner=owner,
                               name=name,
                               private=FASTLANE_DEV in owner):
            raise Exception(f"clone repository({owner}/{name}) failed.")

        patterns = {
            "fastlane-dev/yeoshin-backend-v2/backend":
                (
                    {
                        "*/backend/*/controller/*",
                        "*/backend/*/usecase/*",
                        "*/backend/common/*",
                        "*/entity/*",
                        "*/type/*"
                    },
                    {
                        "*/test"
                    }
                ),
            "fastlane-dev/yeoshin-backend-v2/admin":
                (
                    {
                        "*/admin/*/controller/*",
                        "*/admin/*/usecase/*",
                        "*/admin/common/*",
                        "*/entity/*",
                        "*/type/*"
                    },
                    {
                        "*/test"
                    }
                )
        }

        repo_path = f"/tmp/repo/{owner}_{name}"

        summary, tree, content = await ingest_async(
            repo_path,
            include_patterns=patterns.get(f"{owner}/{name}/{sub_name}", ({}, {}))[0],
            exclude_patterns=patterns.get(f"{owner}/{name}/{sub_name}", ({}, {}))[1]
        )

        return summary, tree, content
    except Exception as e:
        if "Repository not found" in str(e) or "Not Found" in str(e):
            raise ValueError("error:repo_not_found")
        if "Bad credentials" in str(e) or "API rate limit exceeded" in str(e):
            raise ValueError("error:repo_private")
        raise


async def pull_repo(owner: str, name: str, private=False):
    try:
        repo_url = f"https://{GITHUB_TOKEN}@github.com/{owner}/{name}.git" if private \
            else f"https://github.com/{owner}/{name}.git"
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

        repo = Repo.clone_from(url=repo_url, to_path=repo_path)
        print(f"clone repository({owner}/{name}) success. (HEAD: {repo.head.commit.hexsha})")
        return True
    except Exception as e:
        print(f"repository pull failed. {e}")
        return False


"""
    Fix gitingest
    - https://github.com/cyclotruc/gitingest/pull/259 
"""


async def ingest_async(
        source: str,
        max_file_size: int = 10 * 1024 * 1024,  # 10 MB
        include_patterns: Optional[Union[str, Set[str]]] = None,
        exclude_patterns: Optional[Union[str, Set[str]]] = None,
        branch: Optional[str] = None,
        output: Optional[str] = None,
) -> Tuple[str, str, str]:
    """
    Main entry point for ingesting a source and processing its contents.

    This function analyzes a source (URL or local path), clones the corresponding repository (if applicable),
    and processes its files according to the specified query parameters. It returns a summary, a tree-like
    structure of the files, and the content of the files. The results can optionally be written to an output file.

    Parameters
    ----------
    source : str
        The source to analyze, which can be a URL (for a Git repository) or a local directory path.
    max_file_size : int
        Maximum allowed file size for file ingestion. Files larger than this size are ignored, by default
        10*1024*1024 (10 MB).
    include_patterns : Union[str, Set[str]], optional
        Pattern or set of patterns specifying which files to include. If `None`, all files are included.
    exclude_patterns : Union[str, Set[str]], optional
        Pattern or set of patterns specifying which files to exclude. If `None`, no files are excluded.
    branch : str, optional
        The branch to clone and ingest. If `None`, the default branch is used.
    output : str, optional
        File path where the summary and content should be written. If `None`, the results are not written to a file.

    Returns
    -------
    Tuple[str, str, str]
        A tuple containing:
        - A summary string of the analyzed repository or directory.
        - A tree-like string representation of the file structure.
        - The content of the files in the repository or directory.

    Raises
    ------
    TypeError
        If `clone_repo` does not return a coroutine, or if the `source` is of an unsupported type.
    """
    repo_cloned = False

    try:
        parsed_query: ParsedQuery = await parse_query(
            source=source,
            max_file_size=max_file_size,
            from_web=False,
            include_patterns=include_patterns,
            ignore_patterns=exclude_patterns,
        )

        if parsed_query.url:
            selected_branch = branch if branch else parsed_query.branch  # prioritize branch argument
            parsed_query.branch = selected_branch

            clone_config = parsed_query.extact_clone_config()
            clone_coroutine = clone_repo(clone_config)

            if inspect.iscoroutine(clone_coroutine):
                if asyncio.get_event_loop().is_running():
                    await clone_coroutine
                else:
                    asyncio.run(clone_coroutine)
            else:
                raise TypeError("clone_repo did not return a coroutine as expected.")

            repo_cloned = True

        summary, tree, content = ingest_query(parsed_query)

        if output is not None:
            with open(output, "w", encoding="utf-8") as f:
                f.write(tree + "\n" + content)

        return summary, tree, content
    finally:
        # Clean up the temporary directory if it was created
        if repo_cloned:
            shutil.rmtree(TMP_BASE_PATH, ignore_errors=True)


def ingest(
        source: str,
        max_file_size: int = 10 * 1024 * 1024,  # 10 MB
        include_patterns: Optional[Union[str, Set[str]]] = None,
        exclude_patterns: Optional[Union[str, Set[str]]] = None,
        branch: Optional[str] = None,
        output: Optional[str] = None,
) -> Tuple[str, str, str]:
    """
    Synchronous version of ingest_async.

    This function analyzes a source (URL or local path), clones the corresponding repository (if applicable),
    and processes its files according to the specified query parameters. It returns a summary, a tree-like
    structure of the files, and the content of the files. The results can optionally be written to an output file.

    Parameters
    ----------
    source : str
        The source to analyze, which can be a URL (for a Git repository) or a local directory path.
    max_file_size : int
        Maximum allowed file size for file ingestion. Files larger than this size are ignored, by default
        10*1024*1024 (10 MB).
    include_patterns : Union[str, Set[str]], optional
        Pattern or set of patterns specifying which files to include. If `None`, all files are included.
    exclude_patterns : Union[str, Set[str]], optional
        Pattern or set of patterns specifying which files to exclude. If `None`, no files are excluded.
    branch : str, optional
        The branch to clone and ingest. If `None`, the default branch is used.
    output : str, optional
        File path where the summary and content should be written. If `None`, the results are not written to a file.

    Returns
    -------
    Tuple[str, str, str]
        A tuple containing:
        - A summary string of the analyzed repository or directory.
        - A tree-like string representation of the file structure.
        - The content of the files in the repository or directory.

    See Also
    --------
    ingest_async : The asynchronous version of this function.
    """
    return asyncio.run(
        ingest_async(
            source=source,
            max_file_size=max_file_size,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            branch=branch,
            output=output,
        )
    )


def ingest_query(query: ParsedQuery) -> Tuple[str, str, str]:
    """
    Run the ingestion process for a parsed query.

    This is the main entry point for analyzing a codebase directory or single file. It processes the query
    parameters, reads the file or directory content, and generates a summary, directory structure, and file content,
    along with token estimations.

    Parameters
    ----------
    query : ParsedQuery
        The parsed query object containing information about the repository and query parameters.

    Returns
    -------
    Tuple[str, str, str]
        A tuple containing the summary, directory structure, and file contents.

    Raises
    ------
    ValueError
        If the specified path cannot be found or if the file is not a text file.
    """
    subpath = Path(query.subpath.strip("/")).as_posix()
    path = query.local_path / subpath

    apply_gitingest_file(path, query)

    if not path.exists():
        raise ValueError(f"{query.slug} cannot be found")

    if (query.type and query.type == "blob") or query.local_path.is_file():
        # TODO: We do this wrong! We should still check the branch and commit!
        if not path.is_file():
            raise ValueError(f"Path {path} is not a file")

        relative_path = path.relative_to(query.local_path)

        file_node = FileSystemNode(
            name=path.name,
            type=FileSystemNodeType.FILE,
            size=path.stat().st_size,
            file_count=1,
            path_str=str(relative_path),
            path=path,
        )
        return format_single_file(file_node, query)

    root_node = FileSystemNode(
        name=path.name,
        type=FileSystemNodeType.DIRECTORY,
        path_str=str(path.relative_to(query.local_path)),
        path=path,
    )

    stats = FileSystemStats()

    _process_node(
        node=root_node,
        query=query,
        stats=stats,
    )

    return format_directory(root_node, query)


def apply_gitingest_file(path: Path, query: ParsedQuery) -> None:
    """
    Apply the .gitingest file to the query object.

    This function reads the .gitingest file in the specified path and updates the query object with the ignore
    patterns found in the file.

    Parameters
    ----------
    path : Path
        The path of the directory to ingest.
    query : ParsedQuery
        The parsed query object containing information about the repository and query parameters.
        It should have an attribute `ignore_patterns` which is either None or a set of strings.
    """
    path_gitingest = path / ".gitingest"

    if not path_gitingest.is_file():
        return

    try:
        with path_gitingest.open("rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        warnings.warn(f"Invalid TOML in {path_gitingest}: {exc}", UserWarning)
        return

    config_section = data.get("config", {})
    ignore_patterns = config_section.get("ignore_patterns")

    if not ignore_patterns:
        return

    # If a single string is provided, make it a list of one element
    if isinstance(ignore_patterns, str):
        ignore_patterns = [ignore_patterns]

    if not isinstance(ignore_patterns, (list, set)):
        warnings.warn(
            f"Expected a list/set for 'ignore_patterns', got {type(ignore_patterns)} in {path_gitingest}. Skipping.",
            UserWarning,
        )
        return

    # Filter out duplicated patterns
    ignore_patterns = set(ignore_patterns)

    # Filter out any non-string entries
    valid_patterns = {pattern for pattern in ignore_patterns if isinstance(pattern, str)}
    invalid_patterns = ignore_patterns - valid_patterns

    if invalid_patterns:
        warnings.warn(f"Ignore patterns {invalid_patterns} are not strings. Skipping.", UserWarning)

    if not valid_patterns:
        return

    if query.ignore_patterns is None:
        query.ignore_patterns = valid_patterns
    else:
        query.ignore_patterns.update(valid_patterns)

    return


def _process_node(
        node: FileSystemNode,
        query: ParsedQuery,
        stats: FileSystemStats,
) -> None:
    """
    Process a file or directory item within a directory.

    This function handles each file or directory item, checking if it should be included or excluded based on the
    provided patterns. It handles symlinks, directories, and files accordingly.

    Parameters
    ----------
    node : FileSystemNode
        The current directory or file node being processed.
    query : ParsedQuery
        The parsed query object containing information about the repository and query parameters.
    stats : FileSystemStats
        Statistics tracking object for the total file count and size.

    Raises
    ------
    ValueError
        If an unexpected error occurs during processing.
    """

    if limit_exceeded(stats, node.depth):
        return

    for sub_path in node.path.iterdir():

        symlink_path = None
        if sub_path.is_symlink():
            if not _is_safe_symlink(sub_path, query.local_path):
                print(f"Skipping unsafe symlink: {sub_path}")
                continue

            symlink_path = sub_path
            sub_path = sub_path.resolve()

        if sub_path in stats.visited:
            print(f"Skipping already visited path: {sub_path}")
            continue

        stats.visited.add(sub_path)

        if query.ignore_patterns and _should_exclude(sub_path, query.local_path, query.ignore_patterns):
            continue

        if query.include_patterns and not _should_include(sub_path, query.local_path, query.include_patterns):
            continue

        if sub_path.is_file():
            _process_file(path=sub_path, parent_node=node, stats=stats, local_path=query.local_path)
        elif sub_path.is_dir():

            child_directory_node = FileSystemNode(
                name=sub_path.name,
                type=FileSystemNodeType.DIRECTORY,
                path_str=str(sub_path.relative_to(query.local_path)),
                path=sub_path,
                depth=node.depth + 1,
            )

            # rename the subdir to reflect the symlink name
            if symlink_path:
                child_directory_node.name = symlink_path.name
                child_directory_node.path_str = str(symlink_path)

            _process_node(
                node=child_directory_node,
                query=query,
                stats=stats,
            )

            if not child_directory_node.children:
                continue

            node.children.append(child_directory_node)
            node.size += child_directory_node.size
            node.file_count += child_directory_node.file_count
            node.dir_count += 1 + child_directory_node.dir_count

        else:
            raise ValueError(f"Unexpected error: {sub_path} is neither a file nor a directory")

    node.sort_children()


def _process_file(path: Path, parent_node: FileSystemNode, stats: FileSystemStats, local_path: Path) -> None:
    """
    Process a file in the file system.

    This function checks the file's size, increments the statistics, and reads its content.
    If the file size exceeds the maximum allowed, it raises an error.

    Parameters
    ----------
    path : Path
        The full path of the file.
    parent_node : FileSystemNode
        The dictionary to accumulate the results.
    stats : FileSystemStats
        Statistics tracking object for the total file count and size.
    local_path : Path
        The base path of the repository or directory being processed.
    """
    file_size = path.stat().st_size
    if stats.total_size + file_size > MAX_TOTAL_SIZE_BYTES:
        print(f"Skipping file {path}: would exceed total size limit")
        return

    stats.total_files += 1
    stats.total_size += file_size

    if stats.total_files > MAX_FILES:
        print(f"Maximum file limit ({MAX_FILES}) reached")
        return

    child = FileSystemNode(
        name=path.name,
        type=FileSystemNodeType.FILE,
        size=file_size,
        file_count=1,
        path_str=str(path.relative_to(local_path)),
        path=path,
        depth=parent_node.depth + 1,
    )

    parent_node.children.append(child)
    parent_node.size += file_size
    parent_node.file_count += 1


def limit_exceeded(stats: FileSystemStats, depth: int) -> bool:
    """
    Check if any of the traversal limits have been exceeded.

    This function checks if the current traversal has exceeded any of the configured limits:
    maximum directory depth, maximum number of files, or maximum total size in bytes.

    Parameters
    ----------
    stats : FileSystemStats
        Statistics tracking object for the total file count and size.
    depth : int
        The current depth of directory traversal.

    Returns
    -------
    bool
        True if any limit has been exceeded, False otherwise.
    """
    if depth > MAX_DIRECTORY_DEPTH:
        print(f"Maximum depth limit ({MAX_DIRECTORY_DEPTH}) reached")
        return True

    if stats.total_files >= MAX_FILES:
        print(f"Maximum file limit ({MAX_FILES}) reached")
        return True  # TODO: end recursion

    if stats.total_size >= MAX_TOTAL_SIZE_BYTES:
        print(f"Maxumum total size limit ({MAX_TOTAL_SIZE_BYTES / 1024 / 1024:.1f}MB) reached")
        return True  # TODO: end recursion

    return False


try:
    locale.setlocale(locale.LC_ALL, "")
except locale.Error:
    locale.setlocale(locale.LC_ALL, "C")


def _get_encoding_list() -> List[str]:
    """
    Get list of encodings to try, prioritized for the current platform.

    Returns
    -------
    List[str]
        List of encoding names to try in priority order, starting with the
        platform's default encoding followed by common fallback encodings.
    """
    encodings = [locale.getpreferredencoding(), "utf-8", "utf-16", "utf-16le", "utf-8-sig", "latin"]
    if platform.system() == "Windows":
        encodings += ["cp1252", "iso-8859-1"]
    return encodings


def _should_include(path: Path, base_path: Path, include_patterns: Set[str]) -> bool:
    """
    Determine if the given file or directory path matches any of the include patterns.

    This function checks whether the relative path of a file or directory matches any of the specified patterns. If a
    match is found, it returns `True`, indicating that the file or directory should be included in further processing.

    Parameters
    ----------
    path : Path
        The absolute path of the file or directory to check.
    base_path : Path
        The base directory from which the relative path is calculated.
    include_patterns : Set[str]
        A set of patterns to check against the relative path.

    Returns
    -------
    bool
        `True` if the path matches any of the include patterns, `False` otherwise.
    """
    try:
        rel_path = path.relative_to(base_path)
    except ValueError:
        # If path is not under base_path at all
        return False

    rel_str = str(rel_path)
    # if path is a directory, include it by default
    if path.is_dir():
        return True

    for pattern in include_patterns:
        if fnmatch(rel_str, pattern):
            return True
    return False


def _should_exclude(path: Path, base_path: Path, ignore_patterns: Set[str]) -> bool:
    """
    Determine if the given file or directory path matches any of the ignore patterns.

    This function checks whether the relative path of a file or directory matches
    any of the specified ignore patterns. If a match is found, it returns `True`, indicating
    that the file or directory should be excluded from further processing.

    Parameters
    ----------
    path : Path
        The absolute path of the file or directory to check.
    base_path : Path
        The base directory from which the relative path is calculated.
    ignore_patterns : Set[str]
        A set of patterns to check against the relative path.

    Returns
    -------
    bool
        `True` if the path matches any of the ignore patterns, `False` otherwise.
    """
    try:
        rel_path = path.relative_to(base_path)
    except ValueError:
        # If path is not under base_path at all
        return True

    rel_str = str(rel_path)
    for pattern in ignore_patterns:
        if pattern and fnmatch(rel_str, pattern):
            return True
    return False


if __name__ == "__main__":
    import asyncio


    async def main():
        # exists = await check_repo_exists("fastlane-dev", "yeoshin-backend-v2")
        # print(exists)
        # pull_repo("fastlane-dev", "yeoshin-backend-v2")
        summary, tree, content = await ingest_repo("fastlane-dev", "yeoshin-backend-v2", "backend")
        # summary, tree, content = await ingest_repo("HarishChandran3304", "TTG", "")
        # print(summary)
        print(tree)
        # print(content)
        prompt = await generate_prompt("Response in korean.", [], tree, content)
        chat = await initialize_chat(prompt)
        response = await chat.send_message("프로젝트구조설명해줘")
        print(response)


    asyncio.run(main())
