import os
import sys
from pathlib import Path
from typing import Union

import Metashape
from tqdm import tqdm

from helpers import get_all_files


def handle_missing_mask(camera_label):
    print(f"Missing mask for camera: {camera_label}")
    pass


def handle_error(e):
    print(f"Error: {e}")
    pass


def process_frames(data: Path, frames: Path, mask_path: Union[Path or None] = None,
                   use_mask: bool = False, set_size: int = 100, overlap: int = 10):


    # Create Metashape project
    doc = Metashape.Document()
    doc.save(path=str((data / "metashape_project.psx").resolve()))
    doc.read_only = False

    # Create new chunk
    chunk = doc.addChunk()

    # Get and sort all frame paths for current video file
    frames = sorted(get_all_files(frames, "*"))

    if len(frames) == 0:
        print("No frames found")
        return

    # Add all frames to chunk
    print(f"Adding {len(frames)} frames to chunk")
    chunk.addPhotos([str(path) for path in frames])

    print("Loading masks")
    if use_mask:
        for camera in tqdm(chunk.cameras, desc="Loading masks", dynamic_ncols=True):  # Loading masks progress bar
            # Build the mask file path
            mask_file_path = mask_path / (camera.label + '_mask.png')
            mask_file_path = mask_file_path.resolve()

            # Check if the mask file exists
            if os.path.isfile(mask_file_path):
                # Load the mask file
                camera.mask = Metashape.Mask().load(str(mask_file_path))
            else:
                handle_missing_mask(camera.label)

    total_sets = len(frames) // set_size + (len(frames) % set_size > 0)
    match_window = set_size + overlap

    print(f"Total sets: {total_sets}")
    pbar = tqdm(total=total_sets, desc="Processing", dynamic_ncols=True)

    # Process the frames
    for i in range(0, len(frames), set_size):
        # Select only the current matching window for matching photos
        for j, camera in enumerate(chunk.cameras):
            if i <= j < (i + match_window):
                camera.selected = True
            else:
                camera.selected = False

        print(f"Matching photos {i} to {i + match_window}")
        try:
            chunk.matchPhotos(downscale=2, generic_preselection=True, reference_preselection=False)
        except Exception as e:
            handle_error(e)

        # Select only the current set of frames for alignment
        for j, camera in enumerate(chunk.cameras):
            if i <= j < (i + set_size):
                camera.selected = True
            else:
                camera.selected = False

        print(f"Aligning frames {i} to {i + set_size}")
        try:
            chunk.alignCameras()
        except Exception as e:
            handle_error(e)

        pbar.update(1)

    pbar.close()

    # Optimize the alignment of all cameras
    print("Optimizing alignment...")
    chunk.optimizeCameras()

    # Unselect all cameras
    for camera in chunk.cameras:
        camera.selected = False

    # Counter for selected (i.e., unaligned) cameras
    selected_count = 0

    # Iterate over all cameras
    for camera in tqdm(chunk.cameras, desc="Aligning unaligned",
                       dynamic_ncols=True):  # Aligning unaligned cameras progress bar
        if camera.transform:  # camera is aligned
            camera.selected = False
        else:  # camera is not aligned
            camera.selected = True
            selected_count += 1

        # If 100 unaligned cameras are selected, reset and align them
        if selected_count >= 100:
            chunk.resetAlignment()  # reset alignment
            chunk.alignCameras()  # align selected cameras
            selected_count = 0  # reset counter

    # After going through all cameras, align any remaining unaligned cameras
    if selected_count > 0:
        chunk.resetAlignment()  # reset alignment
        chunk.alignCameras()  # align selected cameras

    print("Final optimization...")
    chunk.optimizeCameras()

    doc.save()
