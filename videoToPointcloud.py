"""
# Tree Reconstruction from Video
Before first use:
1) Download and place the Agisoft Metashape 2.02 WHS file from
https://www.agisoft.com/downloads/installer/ into the `bin` folder.

2) Run the following command in the terminal: ```python setup.py```
"""
import datetime
import shutil
import time
from pathlib import Path

from typing import Union

from Tools import irss_media_tools
from Tools.agisoft_metashape import process_frames
from Tools.irss_media_tools import ModelExecutionEngines
from helpers import get_all_files, get_all_subdir, get_leaf_directories, create_flat_folder_name

from loguru import logger


def process_videos(data_path: Union[str, Path], video_path: Union[str, Path], export_path: Union[str, Path],
                   use_mask: bool = False, regenerate: bool = True, drop_ratio: float = 0.5):
    """
    Function to process video_old files from a specified directory for tree reconstruction

    :param data_path: Directory path where data files are located
    :param video_path: Directory path where video_old files are located
    :param use_mask: Boolean flag indicating if sky removal masks should be generated or not
    :param regenerate: Boolean flag indicating if existing files should be regenerated or not
    """

    start_time = time.time()

    data_path = Path(data_path)
    video_path = Path(video_path)

    if not data_path.exists():
        logger.warning(f"The provided path: {data_path} does not exist.")
        return

    if not video_path.exists():
        logger.warning(f"The provided video_old path: {video_path} does not exist.")
        return

    video_files = get_all_files(video_path, "*.mp4", "*.mov", "*.avi", "*.mkv", "*.wmv", "*.flv", "*.webm")

    tools_path = data_path / "tools"
    # export_path = data_path / "export" / video_files[0].name

    temp_path = data_path / "tmp" / export_path.name

    frames_path = temp_path / "frames"
    mask_path = temp_path / "masks"

    for file in video_files:
        print(file)

    if regenerate:
        logger.debug(f"Regenerating files for {video_files[0].name}")
        # If the directory exists, delete it and all its contents
        if temp_path.is_dir():
            shutil.rmtree(temp_path)

        if frames_path.is_dir():
            shutil.rmtree(frames_path)

        if mask_path.is_dir():
            shutil.rmtree(mask_path)

    temp_path.mkdir(parents=True, exist_ok=True)

    media_tools = irss_media_tools.MediaTools(base_dir=tools_path)

    if not frames_path.exists():
        frame_start_time = time.time()

        # Now recreate the directory
        frames_path.mkdir(parents=True, exist_ok=True)

        for video_file in video_files:
            media_tools.extract_frames(video_file,
                                       frames_path,
                                       drop_ratio)  # 0.95 = 95% of the original video_old dropped or from 60 fps to 3 fps

        frame_end_time = time.time()
        frame_elapsed_time = frame_end_time - frame_start_time

        file_count = 0
        for path in frames_path.rglob('*'):
            if path.is_file():
                file_count += 1
        logger.debug(f"Extracted {file_count} frames in {format_elapsed_time(frame_elapsed_time)}")

    if use_mask and not mask_path.exists():
        mask_path.mkdir(parents=True, exist_ok=True)

        media_tools.mask_sky(frames_path, mask_path, ModelExecutionEngines.CUDA)

        # sky_removal_obj = sky_removal.SkyRemoval(base_dir=tools_path)
        # sky_removal_obj.remove_sky(frames_path, mask_path)

    export_path.mkdir(parents=True, exist_ok=True)

    process_frames(data_path, frames_path, export_path, mask_path)

    end_time = time.time()
    frame_elapsed_time = end_time - start_time
    logger.debug(f"Elapsed execution time: {format_elapsed_time(frame_elapsed_time)}")


def format_elapsed_time(elapsed_time):
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    time_format = ""
    if hours > 0:
        time_format += f"{hours} hour{'s' if hours > 1 else ''} "

    if minutes > 0:
        time_format += f"{minutes} minute{'s' if minutes > 1 else ''} "

    time_format += f"{seconds} second{'s' if seconds > 1 else ''}"

    return time_format


if __name__ == '__main__':
    logger.add("file_{time}.log")
    data_dir = Path("Data")
    video_subdir = data_dir / Path("video")

    footage_group_subdirs = get_leaf_directories(video_subdir)

    logger.debug("Processing video files in the following subdirectories:")
    for subdir in footage_group_subdirs:
        logger.debug(subdir.resolve())

    for subdir in footage_group_subdirs:
        logger.debug(f"Processing video files in {subdir.resolve()}")
        export_path = data_dir / "export" / create_flat_folder_name(subdir, video_subdir)
        process_videos(data_dir, subdir, export_path, use_mask=True, regenerate=True, drop_ratio=0.95)
