from typing import List, Union
from pathlib import Path
from tqdm import tqdm
import Metashape
import os

from helpers import get_all_files


def create_or_load_metashape_project(data: Path):
    project_path = data / "metashape_project.psx"
    project_path = project_path.resolve()

    # Create a Metashape document
    doc = Metashape.Document()
    loaded = False

    # If the project file already exists, load it
    if project_path.exists():
        print("Loading existing project")
        doc.open(str(project_path))
        # Assuming that you want to work with the first chunk in the project
        chunk = doc.chunk if len(doc.chunks) > 0 else doc.addChunk()
        loaded = True
    else:
        # Else create and save a new project
        print("Creating new project")
        doc.save(path=str(project_path))
        doc.read_only = False
        chunk = doc.addChunk()

    return doc, chunk, loaded


def add_frames_to_chunk(chunk: Metashape.Chunk, frames: List[Path]):
    print(f"Adding {len(frames)} frames to chunk")
    chunk.addPhotos([str(path) for path in frames])


def load_masks(chunk: Metashape.Chunk, mask_path: Path, use_mask: bool):
    if not use_mask:
        return

    print("Loading masks")
    for camera in tqdm(chunk.cameras, desc="Loading masks", dynamic_ncols=True):
        mask_file_path = mask_path / (camera.label + '_mask.png')
        mask_file_path = mask_file_path.resolve()

        if os.path.isfile(mask_file_path):
            camera.mask = Metashape.Mask().load(str(mask_file_path))
        else:
            handle_missing_mask(camera.label)


def handle_missing_mask(camera_label: str):
    print(f"No mask file found for camera: {camera_label}")


def handle_error(e: Exception):
    print(f"An error occurred: {e}")


def process_frames(data: Path, frames: Path, mask_path: Union[Path, None] = None,
                   use_mask: bool = False, set_size: int = 250):
    frames = sorted(get_all_files(frames, "*"))
    if not frames:
        print("No frames found in the specified path.")
        return

    doc, chunk, loaded = create_or_load_metashape_project(data)

    if len(chunk.cameras) > 0:
        print("Chunk already has cameras, skipping adding frames.")
    else:
        add_frames_to_chunk(chunk, frames)
        load_masks(chunk, mask_path, use_mask)

    doc.save()

    total_sets = len(frames) // set_size + (len(frames) % set_size > 0)
    match_window = set_size * 2

    print(f"Total sets: {total_sets}")
    pbar = tqdm(total=total_sets, desc="Processing", dynamic_ncols=True)

    for i in range(0, len(frames), set_size):
        match_list = list()
        align_list = list()

        for j, camera in enumerate(chunk.cameras):
            if i <= j < (i + match_window):
                match_list.append(camera)
            if i <= j < (i + set_size):
                align_list.append(camera)

        print(f"Matching photos {i} to {i + match_window}")
        try:
            chunk.matchPhotos(downscale=2, generic_preselection=False,
                              reference_preselection=True,
                              reference_preselection_mode=Metashape.ReferencePreselectionMode.ReferencePreselectionSequential)
        except Exception as e:
            handle_error(e)

        print(f"Aligning frames {i} to {i + set_size}")
        try:
            chunk.alignCameras(cameras=align_list)
        except Exception as e:
            handle_error(e)

        doc.save()

        pbar.update(1)

    pbar.close()

    max_iterations = 10
    iteration = 0

    while iteration < max_iterations:
        print("Optimizing alignment...")
        chunk.optimizeCameras()

        unaligned_cameras = 0
        realign_list = list()
        pbar_realign = tqdm(total=len(chunk.cameras), desc="Realigning", dynamic_ncols=True)
        for camera in chunk.cameras:
            pbar_realign.update()
            if not camera.transform:
                realign_list.append(camera)
                unaligned_cameras += 1
            if len(realign_list) >= set_size:
                chunk.alignCameras(reset_alignment=True, cameras=realign_list)
                realign_list = list()
                doc.save()

        pbar_realign.close()

        if realign_list:
            chunk.alignCameras(reset_alignment=True, cameras=realign_list)

        if unaligned_cameras == 0:
            break

        iteration += 1

    if iteration == max_iterations:
        print(f"Stopped realignment after {max_iterations} iterations.")
    else:
        print(f"All cameras realigned after {iteration} iterations.")

    doc.save()

    print("Final optimization...")
    chunk.optimizeCameras()

    doc.save()
