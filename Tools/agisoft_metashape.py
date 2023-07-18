from typing import List, Union
from pathlib import Path

import numpy as np
from Metashape import Document, Chunk, Camera
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

    doc.read_only = False

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


def optimize_cameras(chunk: Chunk):
    chunk.optimizeCameras(fit_f=True, fit_cx=True, fit_cy=True,
                          fit_k1=True, fit_k2=True, fit_k3=True, fit_k4=True)


def match_photos(chunk, set_size: int = 250, overlap_ratio: float = 0.3):
    number_of_cameras = len(chunk.cameras)

    # Calculate the total number of sets by floor dividing the total number of frames by the set size.
    # If there is a remainder, our modulus will be greater than 0, making the bool True.
    # bool True = 1.
    total_sets = (number_of_cameras // set_size) + (number_of_cameras % set_size > 0)

    # Calculate the overlap between sets by multiplying the set size by the overlap ratio.
    overlap = int(set_size * overlap_ratio)

    print(f"Total sets for matching: {total_sets}")

    with tqdm(total=total_sets, desc="Matching", dynamic_ncols=True) as pbar:
        for i in range(0, number_of_cameras, set_size):
            match_list = list()

            start_index = max(0, i - overlap)
            end_index = min(number_of_cameras, i + set_size + overlap)

            for j, camera in enumerate(chunk.cameras[start_index:end_index]):
                match_list.append(camera)

            print(f"Matching photos {start_index} to {end_index}")
            try:
                chunk.matchPhotos(cameras=match_list, downscale=2, generic_preselection=False,
                                  reference_preselection=False, keep_keypoints=True, keypoint_limit=60000)
            except Exception as e:
                handle_error(e)

            pbar.update(1)


def align_cameras(chunk, set_size: int = 250):
    number_of_frames = len(chunk.cameras)

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

            pbar.update(1)


def realign_cameras(doc: Document, chunk: Chunk, batch_size: int = 50, max_iterations: int = 5,
                    max_error: int = 1.1) -> None:
    """
    Tries to realign cameras in the given chunk by attempting to optimize camera alignment multiple times. It stops if the process isn't improving or if it reaches a maximum iteration count.

    Args:
        doc (Document): The Metashape Document.
        chunk (Chunk): The Metashape Chunk.
        batch_size (int, optional): The number of cameras to be realigned per iteration. Defaults to 50.
        max_iterations (int, optional): Maximum number of iterations to perform. Defaults to 5.
        max_error (int, optional): Maximum allowed error in camera alignment. Defaults to 0.1.
            """

    if max_error < 1:
        print(
            f"realign_cameras called with max_error {max_error}. "
            f"Max error must be greater than 1. Setting to {max_error + 1}")
        max_error = max_error + 1

    def iteratively_align(cameras: List[Camera]) -> None:
        for camera in cameras:
            camera.transform = None

        try:
            chunk.alignCameras(cameras=cameras)
        except Exception as error:
            handle_error(error)

    iteration = 0
    stale_iterations = 0

    pbar_iteration = tqdm(total=max_iterations, desc="Realignment Iterations", dynamic_ncols=True)

    num_unaligned_cameras_start = sum(camera.transform is None for camera in chunk.cameras)
    initial_alignment = num_unaligned_cameras_start
    total_cameras = len(chunk.cameras)

    while iteration < max_iterations:
        print(f"Optimizing alignment... Iteration {iteration + 1}")

        realign_batch = []
        for camera in chunk.cameras:
            if camera.transform is None:
                realign_batch.append(camera)
                if len(realign_batch) == batch_size:  # if the batch size is reached, align the cameras
                    iteratively_align(realign_batch)
                    realign_batch = []  # clear the batch
            elif realign_batch:  # if we hit an aligned camera and have unaligned cameras in the batch
                iteratively_align(realign_batch)
                realign_batch = []  # clear the batch

        if realign_batch:  # align any remaining unaligned cameras
            iteratively_align(realign_batch)

        num_unaligned_cameras_end = sum(camera.transform is None for camera in chunk.cameras)

        if num_unaligned_cameras_end == num_unaligned_cameras_start:
            stale_iterations += 1
        else:
            stale_iterations = 0

        if stale_iterations > 1:
            break

        num_unaligned_cameras_start = num_unaligned_cameras_end
        iteration += 1
        pbar_iteration.update()

    pbar_iteration.close()

    if iteration == max_iterations:
        print(f"Stopped realignment after {max_iterations} iterations.")
    else:
        print(f"All cameras realaligned after {iteration} iterations.")

    print("Final optimization...")

    try:
        optimize_cameras(chunk)
    except Exception as error:
        handle_error(error)

    doc.save()

    final_alignment = sum(camera.transform is None for camera in chunk.cameras)

    improvement = (final_alignment - initial_alignment) / total_cameras * 100

    print(f"Cameras realigned. "
          f"Initial alignment: {initial_alignment}."
          f"Final alignment: {final_alignment}."
          f"Improvement {improvement:.2f}% ")


def remove_low_quality_cameras(chunk: Chunk, threshold: float = 0.5) -> None:
    chunk.analyzeImages()

    cameras_to_remove = []

    # Loop through cameras
    for camera in tqdm(chunk.cameras, desc="Scanning for low quality cameras", dynamic_ncols=True):
        quality = float(camera.meta['Image/Quality'])
        if quality < threshold:
            cameras_to_remove.append(camera)

    chunk.remove(cameras_to_remove)


def optimize_alignment(chunk: Metashape.Chunk, upper_percentile: float = 90, lower_percentile: float = 10, iterations: int = 10) -> None:
    """
    Iteratively removes points with reprojection accuracy higher than the calculated threshold.
    """

    points = chunk.tie_points.points
    f = Metashape.TiePoints.Filter()

    # Initialize filter for projection accuracy
    f.init(chunk, criterion=Metashape.TiePoints.Filter.ProjectionAccuracy)

    # Calculate initial and final thresholds based on given percentiles
    accuracies = [p.projection_accuracy for p in points if p.valid]
    accuracy_threshold = np.percentile(accuracies, upper_percentile)
    final_threshold = np.percentile(accuracies, lower_percentile)

    # Calculate decrement based on the difference between initial and final thresholds
    decrement = (accuracy_threshold - final_threshold) / iterations

    for _ in range(iterations):
        # Select points based on the current threshold
        f.selectPoints(accuracy_threshold)

        # Calculate number of selected points
        nselected = len([True for point in points if point.valid and point.selected])

        if nselected == 0:
            print("No more points with high reprojection accuracy.")
            break

        # Remove selected points
        chunk.point_cloud.removeSelectedPoints()

        # Lower the threshold for the next iteration
        accuracy_threshold -= decrement

        optimize_cameras(chunk)




def process_frames(data: Path, frames: Path, mask_path: Union[Path, None] = None,
                   use_mask: bool = False, set_size: int = 250, match: bool = True, align: bool = True,
                   realign: bool = True):
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

    if True:
        remove_low_quality_cameras(chunk)

        doc.save()

        match_photos(chunk, set_size)

        doc.save()

        align_cameras(chunk, set_size)

        doc.save()

    optimize_alignment(chunk)

    doc.save()

    realign_cameras(doc, chunk, set_size)

    doc.save()

    optimize_alignment(chunk)

    doc.save()

    print("Done!")
