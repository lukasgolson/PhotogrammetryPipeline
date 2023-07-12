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

from Tools import IRSSMediaTools
from helpers import get_all_files
import Metashape

# %%
DATA_PATH = Path("data")
VIDEO_PATH = DATA_PATH / "video"
TEMP_PATH = Path("tmp")
# %%
if TEMP_PATH.exists():
    shutil.rmtree(TEMP_PATH)
# %%
video_files = get_all_files(VIDEO_PATH, "*")

for file in video_files:
    print(file)
# %%
for video_file in video_files:
    IRSSMediaTools.extract_frames(video_file, TEMP_PATH / "frames", 0.75)
# %%

for image_file in get_all_files(TEMP_PATH / "frames", "*"):
    print(image_file)
