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
import shutil
from pathlib import Path

import Tools
from Tools.irss_media_tools import MediaTools
from Tools.sky_removal import SkyRemoval
from helpers import get_all_files

# %%
DATA_PATH = Path("data")
VIDEO_PATH = DATA_PATH / "video"
TEMP_PATH = Path("tmp")
FRAMES_PATH = TEMP_PATH / "frames"
MASK_PATH = TEMP_PATH / "mask"

# %%
if TEMP_PATH.exists():
    shutil.rmtree(TEMP_PATH)
# %%
video_files = get_all_files(VIDEO_PATH, "*")

for file in video_files:
    print(file)
# %%


media_tools = MediaTools()
for video_file in video_files:
    media_tools.extract_frames(video_file, FRAMES_PATH, 0.5)
#%%

sky_removal = SkyRemoval()
sky_removal.remove_sky(FRAMES_PATH, MASK_PATH)

#%%
