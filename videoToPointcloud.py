"""
# Tree Reconstruction from Video
Before first use:
1) Download and place the Agisoft Metashape 2.02 WHS file from
https://www.agisoft.com/downloads/installer/ into the `bin` folder.

2) Run the following command in the terminal: ```python setup.py```
"""
import shutil
import time
from pathlib import Path

from typing import Union

from Tools import irss_media_tools
from Tools.agisoft_metashape import process_frames
from Tools.irss_media_tools import ModelExecutionEngines
from helpers import get_all_files


def process_videos(data_path: Union[str, Path], video_path: Union[str, Path],
                   use_mask: bool = False, regenerate: bool = True):
    """
    Function to process video files from a specified directory for tree reconstruction

    :param data_path: Directory path where data files are located
    :param video_path: Directory path where video files are located
    :param use_mask: Boolean flag indicating if sky removal masks should be generated or not
    :param regenerate: Boolean flag indicating if existing files should be regenerated or not
    """

    start_time = time.time()

    data_path = Path(data_path)
    video_path = Path(video_path)

    if not data_path.exists():
        print(f"The provided path: {data_path} does not exist.")
        return

    if not video_path.exists():
        print(f"The provided video path: {video_path} does not exist.")
        return

    tools_path = data_path / "tools"
    temp_path = data_path / "tmp"
    export_path = data_path / "export"

    frames_path = temp_path / "frames"
    mask_path = temp_path / "masks"

    video_files = get_all_files(video_path, "*")

    for file in video_files:
        print(file)

    if regenerate:
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
        # Now recreate the directory
        frames_path.mkdir(parents=True, exist_ok=True)

        for video_file in video_files:
            media_tools.extract_frames(video_file,
                                       frames_path, 0.9)  # 0.75 = 75% of the original video or from 60 fps to 15 fps

    if use_mask and not mask_path.exists():
        mask_path.mkdir(parents=True, exist_ok=True)

        media_tools.mask_sky(frames_path, mask_path, ModelExecutionEngines.CUDA)

        # sky_removal_obj = sky_removal.SkyRemoval(base_dir=tools_path)
        # sky_removal_obj.remove_sky(frames_path, mask_path)

    export_path.mkdir(parents=True, exist_ok=True)

    process_frames(data_path, frames_path, export_path, mask_path, use_mask)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed execution time: {elapsed_time}")


if __name__ == '__main__':
    data_dir = Path("Data")
    video_subdir = data_dir / Path("video")

    process_videos(data_dir, video_subdir, use_mask=True, regenerate=False)
