# %% md
# # Tree Reconstruction from Video
# 
# To install before first use:
# 1) Download and place the Agisoft Metashape 2.02 WHS file from
# https://www.agisoft.com/downloads/installer/ into the `bin` folder.
#
# 2) Download and place the latest release of IRSS Media Tools
# from https://github.com/lukasdragon/MediaTools/releases into the `exec` folder.
#
# 3) run the following command in the terminal: ```python setup.py```
# 
# %%
import shutil
from pathlib import Path

import IRSSMediaTools
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
# Create Metashape project
doc = Metashape.Document()
doc.save(path=str(DATA_PATH / "metashape_project.psx"))

# Create new chunk
chunk = doc.addChunk()
#%%

# Get all frame paths for current video file
frames = get_all_files(TEMP_PATH / "frames", "*")

# Add frames to chunk
chunk.addPhotos([str(path) for path in frames])

# %%
# Matching photos
# downscale ratio 0 = highest, high = 1, medium = 2, low = 4, lowest = 8
chunk.matchPhotos(downscale=2, generic_preselection=True, reference_preselection=False)
chunk.alignCameras()
doc.save()
# %%

# Build dense cloud
chunk.buildDepthMaps(quality=Metashape.HighQuality)
chunk.buildDenseCloud()

# Build model
chunk.buildModel(surface=Metashape.Arbitrary, interpolation=Metashape.EnabledInterpolation)

# Build texture
chunk.buildTexture(blending=Metashape.MosaicBlending, size=4096)

# Save the project
# %%
# Export the dense point cloud
output_path = DATA_PATH / "output"
output_path.mkdir(parents=True, exist_ok=True)

for i, chunk in enumerate(doc.chunks):
    dense_cloud_path = output_path / f"dense_cloud_{i}.ply"  # adjust the extension based on the format you want
    chunk.exportPoints(str(dense_cloud_path), source_data=Metashape.DenseCloudData)
