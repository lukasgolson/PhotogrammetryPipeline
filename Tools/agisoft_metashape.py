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


def load_masks(chunk: Metashape.Chunk, mask_path: Path):
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


def matching_stage(doc, chunk, number_of_frames: int, set_size: int = 250, overlap_ratio: float = 0.3):
    # Calculate the total number of sets by floor dividing the total number of frames by the set size.
    # If there is a remainder, our modulus will be greater than 0, making the bool True.
    # bool True = 1.
    total_sets = (number_of_frames // set_size) + (number_of_frames % set_size > 0)

    # Calculate the overlap between sets by multiplying the set size by the overlap ratio.
    overlap = int(set_size * overlap_ratio)

    print(f"Total sets for matching: {total_sets}")

    with tqdm(total=total_sets, desc="Matching", dynamic_ncols=True) as pbar:
        for i in range(0, number_of_frames, set_size):
            match_list = list()

            start_index = max(0, i - overlap)
            end_index = min(number_of_frames, i + set_size + overlap)

            for j, camera in enumerate(chunk.cameras[start_index:end_index]):
                match_list.append(camera)

            print(f"Matching photos {start_index} to {end_index}")
            try:
                chunk.matchPhotos(cameras=match_list, downscale=2, generic_preselection=False)
            except Exception as e:
                handle_error(e)

            doc.save()
            pbar.update(1)


def alignment_stage(doc, chunk, number_of_frames: int, set_size: int = 250):
    total_sets = number_of_frames // set_size + (number_of_frames % set_size > 0)

    print(f"Total sets for alignment: {total_sets}")

    with tqdm(total=total_sets, desc="Aligning", dynamic_ncols=True) as pbar:
        for i in range(0, number_of_frames, set_size):
            align_list = list()

            for j, camera in enumerate(chunk.cameras):
                if i <= j < (i + set_size):
                    align_list.append(camera)

            print(f"Aligning frames {i} to {i + set_size}")
            try:
                chunk.alignCameras(cameras=align_list)
            except Exception as e:
                handle_error(e)

            doc.save()
            pbar.update(1)

    return doc, chunk


def realignment_phase(doc, chunk, set_size: int = 50, max_iterations: int = 5):
    iteration = 0

    def align_cameras(cameras):
        try:
            chunk.alignCameras(reset_alignment=True, cameras=cameras)
            doc.save()
        except Exception as e:
            handle_error(e)

    while iteration < max_iterations:
        print(f"Optimizing alignment... Iteration {iteration + 1}")
        try:
            chunk.optimizeCameras()
            doc.save()
        except Exception as e:
            handle_error(e)

        realign_list = list()
        pbar_realign = tqdm(total=len(chunk.cameras), desc="Realigning", dynamic_ncols=True)

        for camera in chunk.cameras:
            if camera.transform is None:
                realign_list.append(camera)

                if len(realign_list) >= set_size:
                    align_cameras(realign_list)
                    realign_list.clear()

            elif realign_list:
                align_cameras(realign_list)
                realign_list.clear()

            pbar_realign.update()

        pbar_realign.close()

        # In case any cameras are left unprocessed
        if realign_list:
            align_cameras(realign_list)

        if not any(camera.transform is None for camera in chunk.cameras):
            break

        iteration += 1

    if iteration == max_iterations:
        print(f"Stopped realignment after {max_iterations} iterations.")
    else:
        print(f"All cameras realigned after {iteration} iterations.")

    doc.save()

    print("Final optimization...")
    try:
        chunk.optimizeCameras()
        doc.save()
    except Exception as e:
        handle_error(e)


def process_frames(data: Path, frames: Path, mask_path: Union[Path, None] = None,
                   use_mask: bool = False, set_size: int = 250):
    doc, chunk, loaded = create_or_load_metashape_project(data)

    if len(chunk.cameras) > 0:
        print("Chunk already has cameras, skipping adding frames.")
    else:
        frame_list = sorted(get_all_files(frames, "*"))
        if not frame_list:
            print("No frames found in the specified path.")
            return

        add_frames_to_chunk(chunk, frame_list)

        if use_mask:
            load_masks(chunk, mask_path)

    doc.save()

    if doc is None or chunk is None:
        return

    matching_stage(doc, chunk, frames, set_size)

    # chunk.matchPhotos(downscale=2, generic_preselection=False, )

    alignment_stage(doc, chunk, len(frame_list), set_size)
    realignment_phase(doc, chunk, set_size)

    print("Done!")
