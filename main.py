# %% md
# # Tree Reconstruction from Video
# 
# To install before first use:
# 1) Download and place the Agisoft Metashape 2.02 WHS file from
# https://www.agisoft.com/downloads/installer/ into the `bin` folder.
#
#
# 2) run the following command in the terminal: ```python setup.py```
# 
# %%
import os
import shutil
import sys
from pathlib import Path

from Tools.irss_media_tools import MediaTools
from Tools.sky_removal import SkyRemoval
from agisoft_metashape import process_frames
from helpers import get_all_files


def process_videos(data: Path, video: Path, mask: bool):
    data = Path(data)
    video = Path(video)

    frames_path = data / "frames"
    mask_path = data / "mask"
    tools_path = data / "tools"

    # %%
    video_files = get_all_files(video, "*")

    for file in video_files:
        print(file)
    # %%

    media_tools = MediaTools(base_dir=tools_path)
    for video_file in video_files:
        media_tools.extract_frames(video_file, frames_path, 0.5)
    # %%

    if mask:
        sky_removal = SkyRemoval(base_dir=tools_path)
        sky_removal.remove_sky(frames_path, mask_path)

    # %%

    process_frames(data, frames_path)


if __name__ == '__main__':
    video_path = "data/videos"
    data_path = "data"

    process_videos(data_path, video_path, False)
