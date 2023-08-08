from pathlib import Path
from typing import List, Union

import Metashape
import numpy as np
from Metashape import Chunk, Camera
from tqdm import tqdm

from helpers import get_all_files


class TqdmUpdate(tqdm):
    def update_to(self, p=1):
        self.update(p - self.n)  # Provide here progress increment
        if p >= self.total:  # Check if the progress is complete
            self.close()


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
        # Assuming that we'd want to work with the first chunk in the project
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
    mask_file_path = str(mask_path.resolve()) + "/{filename}_mask.png"

    load_masks_loading_bar = TqdmUpdate(total=100, desc="Loading Masks")

    try:
        chunk.generateMasks(path=mask_file_path, masking_mode=Metashape.MaskingModeFile,
                            progress=load_masks_loading_bar.update_to)



    except Exception as e:
        print(f"An error occurred while generating masks: {e}. Continuing...")


def handle_missing_mask(camera_label: str):
    print(f"No mask file found for camera: {camera_label}")


def handle_error(e: Exception):
    print(f"An error occurred: {e}")


def optimize_cameras(chunk: Chunk):
    chunk.optimizeCameras(fit_corrections=True)


def iterative_match_photos(chunk, set_size: int = 250, overlap_ratio: float = 0.3):
    number_of_cameras = len(chunk.cameras)

    # Calculate the total number of sets by floor dividing the total number of frames by the set size.
    # If there is a remainder, our modulus will be greater than 0, making the bool True,
    # adding an additional set (bool True = 1).
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

            matching_bar = TqdmUpdate(total=100, desc=f"Matching images, iteration: {i} ")

            print(f"Matching photos {start_index} to {end_index}")
            try:
                chunk.matchPhotos(cameras=match_list, downscale=1, generic_preselection=False,
                                  reference_preselection=False, keep_keypoints=True, keypoint_limit=40000,
                                  tiepoint_limit=4000, progress=matching_bar.update_to)
            except Exception as e:
                handle_error(e)

            pbar.update(1)


def align_cameras(chunk: Chunk, cameras: List[Camera]) -> None:
    for camera in cameras:
        camera.transform = None
    try:
        chunk.alignCameras(cameras=cameras)
    except Exception as error:
        handle_error(error)


def iterative_align_cameras(chunk: Chunk, batch_size: int = 50, max_iterations: int = 5, timeout=2) -> None:
    """
    Tries to realign cameras in the given chunk by attempting to optimize camera alignment multiple times.
    It stops if the process isn't improving as set by the timeout or if it reaches a maximum iteration count.

    :param chunk: The Metashape Chunk.
    :param batch_size: The number of cameras to be realigned per iteration. Defaults to 50.
    :param max_iterations: Maximum number of iterations to perform. Defaults to 5.
    :param timeout: Maximum number of stale iterations to perform before finishing. Defaults to 2.
    """

    iteration = 0
    stale_iterations = 0

    num_unaligned_cameras_start = sum(camera.transform is None for camera in chunk.cameras)
    initial_alignment = num_unaligned_cameras_start
    total_cameras = len(chunk.cameras)

    pbar_iteration = tqdm(total=max_iterations, desc="Realignment Iterations", dynamic_ncols=True)

    while iteration < max_iterations:
        realign_batch = []
        for camera in tqdm(chunk.cameras, desc=f"Iteratively aligning... Iteration {iteration + 1}", dynamic_ncols=True):
            if camera.transform is None:
                realign_batch.append(camera)
                if len(realign_batch) == batch_size:  # if the batch size is reached, align the cameras
                    align_cameras(chunk, realign_batch)
                    realign_batch = []  # clear the batch
            elif realign_batch:  # if we hit an aligned camera and have unaligned cameras in the batch
                align_cameras(chunk, realign_batch)
                realign_batch = []  # clear the batch

        if realign_batch:  # align any remaining unaligned cameras
            align_cameras(chunk, realign_batch)

        num_unaligned_cameras_end = sum(camera.transform is None for camera in chunk.cameras)

        if num_unaligned_cameras_end == num_unaligned_cameras_start:
            stale_iterations += 1
        else:
            stale_iterations = 0

        if stale_iterations >= timeout:
            break

        if iteration == max_iterations:
            print(f"Stopped realignment after {max_iterations} iterations.")
        else:
            print(f"All cameras realaligned after {iteration} iterations.")

        num_unaligned_cameras_start = num_unaligned_cameras_end
        iteration += 1
        pbar_iteration.update()

        pbar_iteration.close()

        final_alignment = sum(camera.transform is None for camera in chunk.cameras)

        improvement = (final_alignment - initial_alignment) / total_cameras * 100

        print(f"Cameras realigned. "
              f"Initial alignment: {initial_alignment}."
              f"Final alignment: {final_alignment}."
              f"Improvement {improvement:.2f}% ")


def remove_low_quality_cameras(chunk: Chunk, threshold: float = 0.5) -> None:
    chunk.analyzeImages(filter_mask=True)

    cameras_to_remove = []

    # Loop through cameras
    for camera in tqdm(chunk.cameras, desc="Scanning for low quality cameras", dynamic_ncols=True):
        quality = float(camera.meta['Image/Quality'])
        if quality < threshold:
            cameras_to_remove.append(camera)

    print(f"Removing {len(cameras_to_remove)} low quality cameras")
    chunk.remove(cameras_to_remove)


def optimize_alignment(chunk: Metashape.Chunk, upper_percentile: float = 90, lower_percentile: float = 10,
                       delta_ratio: float = 0.1, total_allowable_error: float = 0.5, iterations: int = 10) -> None:
    """
        Iteratively optimizes the alignment of tie points in a Metashape Chunk. This is accomplished by removing points
        with high reprojection error and adjusting camera positions. The iterative process continues until the mean or
        delta mean reprojection error falls below a specified threshold or the maximum number of iterations is reached.

        :param chunk: The Metashape Chunk to be optimized.
        :param upper_percentile: Percentile of the reprojection errors used to set the initial threshold for point removal.
        :param lower_percentile: Percentile of the reprojection errors used to set the final threshold for point removal.
        :param delta_ratio: Ratio used to calculate the error delta from the mean error. This controls how much the mean
                        error needs to decrease in each iteration for the process to continue. A larger delta_ratio
                        tolerates larger changes in mean error, potentially leading to more iterations but possibly less
                        precise optimization. A smaller delta_ratio requires smaller changes in mean error, potentially
                        leading to fewer iterations but more precise optimization. Adjust based on computational resources
                        and desired precision.
        :param total_allowable_error: Maximum reprojection error allowed. If the mean error falls below this value, the optimization process is stopped.
        :param iterations: Maximum number of iterations for the optimization process.
    """

    points = chunk.tie_points.points

    if chunk.tie_points.points is None:
        print("No tie points found")
        return

    tie_point_filter_error = Metashape.TiePoints.Filter()

    tie_point_filter_error.init(chunk, criterion=Metashape.TiePoints.Filter.ReprojectionError)

    reprojection_errors = [tie_point_filter_error.values[i] for i in range(len(points)) if points[i].valid]

    threshold = np.percentile(reprojection_errors, upper_percentile)
    final_threshold = np.percentile(reprojection_errors, lower_percentile)

    decrement = (threshold - final_threshold) / iterations

    old_mean_error = np.mean(reprojection_errors)

    error_delta = old_mean_error * delta_ratio

    for _ in tqdm(range(iterations), desc="Optimizing alignment"):
        tie_point_filter_error.selectPoints(threshold)
        chunk.tie_points.removeSelectedPoints()

        threshold -= decrement
        optimize_cameras(chunk)

        reprojection_errors = [tie_point_filter_error.values[i] for i in range(len(points)) if points[i].valid]
        new_mean_error = np.mean(reprojection_errors)

        error = abs(new_mean_error - old_mean_error)
        if error < error_delta or new_mean_error < total_allowable_error:
            break

        old_mean_error = new_mean_error


def filter_reconstruction_uncertainty(chunk: Chunk, threshold: float = 75):
    # 75 based on trial and error; supported by https://doi.org/10.1007/s00468-019-01866-x
    tie_point_filter_uncertainty = Metashape.TiePoints.Filter()
    tie_point_filter_uncertainty.init(chunk, criterion=Metashape.TiePoints.Filter.ReconstructionUncertainty)

    tie_point_filter_uncertainty.selectPoints(threshold)
    chunk.tie_points.removeSelectedPoints()
    chunk.optimizeCameras(fit_corrections=True)


def filter_projection_accuracy(chunk: Chunk, threshold: float = 10):
    tie_point_filter_accuracy = Metashape.TiePoints.Filter()
    tie_point_filter_accuracy.init(chunk, criterion=Metashape.TiePoints.Filter.ProjectionAccuracy)
    tie_point_filter_accuracy.selectPoints(threshold)
    chunk.tie_points.removeSelectedPoints()
    chunk.optimizeCameras(fit_corrections=True)


def save(doc: Metashape.Document):
    if not doc.read_only:
        doc.save()


def reduce_cameras(chunk) -> None:
    return None


def process_frames(data: Path, frames: Path, export: Path, mask_path: Union[Path, None] = None,
                   use_mask: bool = False, set_size: int = None):
    doc, chunk, loaded = create_or_load_metashape_project(export)

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

    save(doc)

    if doc is None or chunk is None:
        return

    remove_low_quality_cameras(chunk)

    save(doc)

    if set_size is None:
        set_size = int(len(chunk.cameras) * 0.1)

    print("Initial matching and alignment of photos at low resolution.")
    chunk.matchPhotos(cameras=chunk.cameras, downscale=1, generic_preselection=True, reference_preselection=False,
                      reset_matches=True, keep_keypoints=True,
                      keypoint_limit=40000, tiepoint_limit=4000)

    chunk.alignCameras(cameras=chunk.cameras, adaptive_fitting=True, reset_alignment=True)

    save(doc)

    iterative_align_cameras(chunk, set_size)

    save(doc)

    optimize_alignment(chunk)

    filter_reconstruction_uncertainty(chunk)

    filter_projection_accuracy(chunk)

    save(doc)

    iterative_align_cameras(chunk, set_size)

    save(doc)

    build_rough_model_progress_bar = TqdmUpdate(total=100, desc="Building Rough Model")

    chunk.buildModel(surface_type=Metashape.SurfaceType.Arbitrary, source_data=Metashape.DataSource.TiePointsData,
                     progress=build_rough_model_progress_bar.update_to)

    save(doc)

    reduce_overlap_progress_bar = TqdmUpdate(total=100, desc="Reducing camera overlap")

    chunk.reduceOverlap(overlap=3, progress=reduce_overlap_progress_bar.update_to)

    save(doc)

    chunk.generateMasks(masking_mode=Metashape.MaskingModeModel, mask_operation=Metashape.MaskOperationReplacement)
    save(doc)

    build_depths_progress_bar = TqdmUpdate(total=100, desc="Building initial depth maps")

    # build depth map in ultra quality
    chunk.buildDepthMaps(downscale=1, filter_mode=Metashape.MildFiltering, max_neighbors=100,
                         cameras=chunk.cameras, progress=build_depths_progress_bar.update_to)

    save(doc)

    build_point_cloud_progress_bar = TqdmUpdate(total=100, desc="Building point cloud")

    chunk.buildPointCloud(source_data=Metashape.DataSource.DepthMapsData, point_confidence=True, keep_depth=True,
                          progress=build_point_cloud_progress_bar.update_to)

    save(doc)

    chunk.exportPointCloud(str((export / "pointcloud.ply").resolve()), source_data=Metashape.DataSource.PointCloudData)

    save(doc)

    print("Done!")
