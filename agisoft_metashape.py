import sys
from pathlib import Path

import Metashape

from helpers import get_all_files


def process_frames(data: Path, frames: Path, set_size: int = 100):
    # %%
    # Create Metashape project
    doc = Metashape.Document()

    doc.save(path="metashape_project.psx")

    # Create new chunk
    chunk = doc.addChunk()

    # Get all frame paths for current video file
    frames = get_all_files(frames, "*")

    # Add all frames to chunk
    chunk.addPhotos([str(path) for path in frames])

    # Set the reference frame
    p = chunk.cameras[0].mask

    doc.save()

    # Divide frames into chunks of 100
    chunk_size = 100

    # Process the frames in sets
    for i in range(0, len(frames), set_size):
        # Select only the current set of X frames for matching and alignment
        for j, camera in enumerate(chunk.cameras):
            if i <= j < (i + set_size):
                camera.selected = True
            else:
                camera.selected = False
        # Matching photos
        chunk.matchPhotos(downscale=2, generic_preselection=True, reference_preselection=False)
        chunk.alignCameras()
        doc.save()

    # enable all cameras
    for camera in chunk.cameras:
        camera.selected = True

    doc.save()


if __name__ == '__main__':
    frames_path = sys.argv[1]
    data_path = sys.argv[2]

    process_frames(Path(data_path), Path(frames_path))

# Build dense cloud
# chunk.buildDepthMaps(quality=Metashape.HighQuality)
# chunk.buildDenseCloud()

# Build model
# chunk.buildModel(surface=Metashape.Arbitrary, interpolation=Metashape.EnabledInterpolation)

# Build texture
# chunk.buildTexture(blending=Metashape.MosaicBlending, size=4096)

# Save the project
# %%
# Export the dense point cloud
# output_path = DATA_PATH / "output"
# output_path.mkdir(parents=True, exist_ok=True)

# for i, chunk in enumerate(doc.chunks):
# dense_cloud_path = output_path / f"dense_cloud_{i}.ply"  # adjust the extension based on the format you want
# chunk.exportPoints(str(dense_cloud_path), source_data=Metashape.DenseCloudData)
