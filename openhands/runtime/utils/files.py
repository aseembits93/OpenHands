import os
from pathlib import Path

from openhands.events.observation import (
    ErrorObservation,
    FileReadObservation,
    FileWriteObservation,
    Observation,
)


def resolve_path(
    file_path: str,
    working_directory: str,
    workspace_base: str,
    workspace_mount_path_in_sandbox: str,
) -> Path:
    """Resolve a file path to a path on the host filesystem.

    Args:
        file_path: The path to resolve.
        working_directory: The working directory of the agent.
        workspace_mount_path_in_sandbox: The path to the workspace inside the sandbox.
        workspace_base: The base path of the workspace on the host filesystem.

    Returns:
        The resolved path on the host filesystem.
    """
    # Avoid unnecessary Path object creation
    working_directory_path = Path(working_directory)
    workspace_mount_path_in_sandbox_path = Path(workspace_mount_path_in_sandbox)
    workspace_base_path = Path(workspace_base)

    # Apply working directory (fast path: file_path is absolute)
    path_in_sandbox = Path(file_path)
    if not path_in_sandbox.is_absolute():
        path_in_sandbox = working_directory_path / path_in_sandbox

    # Use strict=False to avoid unnecessary filesystem checks in resolve
    abs_path_in_sandbox = path_in_sandbox.resolve(strict=False)
    
    # Precompute parts for startswith optimization
    sandbox_root_parts = workspace_mount_path_in_sandbox_path.parts
    candidate_parts = abs_path_in_sandbox.parts
    len_sandbox_root = len(sandbox_root_parts)
    
    # Fast prefix check for is_relative_to (Python <3.9 compatibility, also generally faster than method call)
    if candidate_parts[:len_sandbox_root] != sandbox_root_parts:
        raise PermissionError(f'File access not permitted: {file_path}')
    
    # Simple tuple slicing is faster than .relative_to for known prefix
    path_in_workspace = Path(*candidate_parts[len_sandbox_root:])
    
    # Host path
    path_in_host_workspace = workspace_base_path / path_in_workspace

    return path_in_host_workspace


def read_lines(all_lines: list[str], start: int = 0, end: int = -1) -> list[str]:
    start = max(start, 0)
    start = min(start, len(all_lines))
    end = -1 if end == -1 else max(end, 0)
    end = min(end, len(all_lines))
    if end == -1:
        if start == 0:
            return all_lines
        else:
            return all_lines[start:]
    else:
        num_lines = len(all_lines)
        begin = max(0, min(start, num_lines - 2))
        end = -1 if end > num_lines else max(begin + 1, end)
        return all_lines[begin:end]


async def read_file(
    path: str,
    workdir: str,
    workspace_base: str,
    workspace_mount_path_in_sandbox: str,
    start: int = 0,
    end: int = -1,
) -> Observation:
    try:
        whole_path = resolve_path(
            path, workdir, workspace_base, workspace_mount_path_in_sandbox
        )
    except PermissionError:
        return ErrorObservation(
            f"You're not allowed to access this path: {path}. You can only access paths inside the workspace."
        )

    try:
        with open(whole_path, 'r', encoding='utf-8') as file:
            lines = read_lines(file.readlines(), start, end)
    except FileNotFoundError:
        return ErrorObservation(f'File not found: {path}')
    except UnicodeDecodeError:
        return ErrorObservation(f'File could not be decoded as utf-8: {path}')
    except IsADirectoryError:
        return ErrorObservation(f'Path is a directory: {path}. You can only read files')
    code_view = ''.join(lines)
    return FileReadObservation(path=path, content=code_view)


def insert_lines(
    to_insert: list[str], original: list[str], start: int = 0, end: int = -1
) -> list[str]:
    """Insert the new content to the original content based on start and end"""
    new_lines = [''] if start == 0 else original[:start]
    new_lines += [i + '\n' for i in to_insert]
    new_lines += [''] if end == -1 else original[end:]
    return new_lines


async def write_file(
    path: str,
    workdir: str,
    workspace_base: str,
    workspace_mount_path_in_sandbox: str,
    content: str,
    start: int = 0,
    end: int = -1,
) -> Observation:
    insert = content.split('\n')

    try:
        whole_path = resolve_path(
            path, workdir, workspace_base, workspace_mount_path_in_sandbox
        )
        if not os.path.exists(os.path.dirname(whole_path)):
            os.makedirs(os.path.dirname(whole_path))
        mode = 'w' if not os.path.exists(whole_path) else 'r+'
        try:
            with open(whole_path, mode, encoding='utf-8') as file:
                if mode != 'w':
                    all_lines = file.readlines()
                    new_file = insert_lines(insert, all_lines, start, end)
                else:
                    new_file = [i + '\n' for i in insert]

                file.seek(0)
                file.writelines(new_file)
                file.truncate()
        except FileNotFoundError:
            return ErrorObservation(f'File not found: {path}')
        except IsADirectoryError:
            return ErrorObservation(
                f'Path is a directory: {path}. You can only write to files'
            )
        except UnicodeDecodeError:
            return ErrorObservation(f'File could not be decoded as utf-8: {path}')
    except PermissionError as e:
        return ErrorObservation(f'Permission error on {path}: {e}')
    return FileWriteObservation(content='', path=path)
