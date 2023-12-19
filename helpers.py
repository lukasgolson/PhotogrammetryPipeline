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


def get_all_subdirs(dir_path: Union[Path, str]) -> List[Path]:
    """
    Retrieve all subdirectories in the specified directory.
    :param dir_path: Directory path as a string or Path object
    :return: List of Path objects of all subdirectories
    """

    dir_path = Path(dir_path) if isinstance(dir_path, str) else dir_path
    subdirs = [p for p in dir_path.iterdir() if p.is_dir()]

    return subdirs
