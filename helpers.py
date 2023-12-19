import re
from pathlib import Path
from typing import Union, List


def get_all_files(dir_path: Union[Path, str], patterns: Union[str, List[str]]) -> List[Path]:
    """
    Retrieve all files in the specified directory that match the given pattern(s).
    :param dir_path: Directory path as a string or Path object
    :param patterns: String or list of strings representing file patterns
    :return: List of Path objects of matching files
    """

    dir_path = Path(dir_path) if isinstance(dir_path, str) else dir_path
    patterns = [patterns] if isinstance(patterns, str) else patterns

    matching_files = {p for pattern in patterns for p in dir_path.rglob(pattern)}

    return list(matching_files)


def get_all_subdir(dir_path: Union[Path, str]) -> List[Path]:
    """
    Retrieve all subdirectories in the specified directory.
    :param dir_path: Directory path as a string or Path object
    :return: List of Path objects of all subdirectories
    """

    dir_path = Path(dir_path) if isinstance(dir_path, str) else dir_path
    subdirs = [p for p in dir_path.iterdir() if p.is_dir()]

    return subdirs


def get_leaf_directories(dir_path: Union[Path, str]) -> List[Path]:
    """
    Retrieve all leaf directories (directories without subdirectories) in the specified directory.
    :param dir_path: Directory path as a string or Path object
    :return: List of Path objects of all leaf directories
    """

    dir_path = Path(dir_path) if isinstance(dir_path, str) else dir_path
    leaf_directories = []

    for p in dir_path.rglob('*'):
        if p.is_dir() and not any(c.is_dir() for c in p.iterdir()):
            leaf_directories.append(p)

    return leaf_directories


def create_flat_folder_name(dir_path: Union[Path, str], base_dir: Union[Path, str]) -> str:
    """
    Create a valid folder name for a flat version of the leaf directory's path.
    :param dir_path: The leaf directory path as a string or Path object
    :param base_dir: The base directory path as a string or Path object
    :return: A string representing the flattened folder name
    """

    dir_path = Path(dir_path) if isinstance(dir_path, str) else dir_path
    base_dir = Path(base_dir) if isinstance(base_dir, str) else base_dir

    relative_path = dir_path.relative_to(base_dir)

    flat_name = re.sub(r'[\\/:*?"<>|]', '_', str(relative_path))

    return flat_name
