import shutil
from pathlib import Path
from typing import List, Union, Tuple

import Metashape
import numpy as np
from Metashape import Chunk, Camera
from loguru import logger
from statemachine import State
from tqdm import tqdm

from Tools.SM import SerializableStateMachine
from helpers import get_all_files


class TqdmUpdate(tqdm):
    def update_to(self, p=1):
        self.update(p - self.n)  # Provide here progress increment
        if p >= self.total:  # Check if the progress is complete
            self.close()


class MetashapeMachine(SerializableStateMachine):

    def __init__(self, data_path: Path, frames_path: Path,
                 mask_path: Union[Path, None] = None,
                 export: [Path, None] = None, set_size_ratio: float = 0.1):

        self.Doc: [Metashape.Document, None] = None
        self.Chunk: [Metashape.Chunk, None] = None
        self.Loaded: bool = False
        self.data_path: Path = data_path
        self.frames_path: Path = frames_path
        self.mask_path: Path = mask_path

        if export is None:
            self.export_path: Path = data_path / "export"
        else:
            self.export_path: Path = export

        self.set_size: int = 128
        self.set_size_ratio: float = set_size_ratio
        self.alignment_count: int = 0

        super().__init__(filename=self.export_path / "sm.pickle")

    load_project = State(initial=True)
    add_frames = State()
    load_sky_masks = State()
    clean_cameras = State(name="Remove low quality images")
    broad_match_photos = State()
    broad_align_cameras = State()
    iterative_align = State()
    optimize_alignment = State(name="Optimize alignment")
    filter_uncertainty = State(name="Filter reconstruction uncertainty")
    filter_accuracy = State(name="Filter projection accuracy")
    second_iterative_align = State(name="Re-filter projection accuracy")
    build_rough_model = State()
    reduce_overlap = State(name="Reduce camera overlap")
    generate_masks = State(name="Generate model-based mask")
    build_depth_maps = State()
    build_point_cloud = State()
    export_point_cloud = State(final=True)

    cycle = (
            load_project.to(add_frames)
            | add_frames.to(load_sky_masks, cond="guard_project_loaded")
            | load_sky_masks.to(clean_cameras)
            | clean_cameras.to(broad_match_photos)
            | broad_match_photos.to(broad_align_cameras)
            | broad_align_cameras.to(iterative_align)
            | iterative_align.to(optimize_alignment)
            | optimize_alignment.to(filter_uncertainty)
            | filter_uncertainty.to(filter_accuracy)
            | filter_accuracy.to(second_iterative_align)
            | second_iterative_align.to(build_rough_model)
            | build_rough_model.to(reduce_overlap)
            | reduce_overlap.to(generate_masks)
            | generate_masks.to(build_depth_maps)
            | build_depth_maps.to(build_point_cloud)
            | build_point_cloud.to(export_point_cloud)
    )

    def guard_project_loaded(self):
        return Chunk is not None

    def guard_prealigned(self):
        return self.alignment_count >= 2

    def initial_state(self):
        return self.load_project  # Start from the initial state

    def get_supplementary_state(self) -> dict:
        dict = {"setSize": self.set_size,
                "setSizeRatio": self.set_size_ratio,
                "alignmentCount": self.alignment_count}
        return dict

    def set_supplementary_state(self, dictionary: dict):
        self.set_size: int = dictionary.get("setSize")
        self.set_size_ratio: dictionary.get("setSizeRatio")
        self.alignment_count: dictionary.get("alignmentCount")
        return True;

    def after_cycle(self):
        if self.Doc is not None and self.Doc.read_only is not True:
            self.Doc.save()

        self.serialize_statemachine()

    def before_cycle(self, event: str, source: State, target: State, message: str = ""):
        message = ". " + message if message else ""
        return f"Transitioning {event} from {source.id} to {target.id}{message}"

    def on_enter_load_project(self):
        logger.info("Loading project.")
        self.Doc, self.Chunk, self.Loaded = create_or_load_metashape_project(self.export_path)

    def on_enter_add_frames(self):
        logger.info("Adding frames.")
        if len(self.Chunk.cameras) > 0:
            logger.debug("Chunk already has cameras, skipping adding frames.")
        else:
            frame_list = sorted(get_all_files(self.frames_path, "*"))
            if not frame_list:
                logger.warning("No frames found in the specified path.")
                return

            add_frames_to_chunk(self.Chunk, frame_list)

        self.set_size = int(len(self.Chunk.cameras)) * self.set_size_ratio

    def on_enter_load_sky_masks(self):
        logger.info("Loading sky masks.")

        if self.mask_path is not None:
            load_masks(self.Chunk, self.mask_path)

    def on_enter_clean_cameras(self):
        logger.info("Cleaning cameras.")
        remove_low_quality_cameras(self.Chunk)

    def on_enter_broad_match_photos(self):
        broad_match_photos_progress_bar = TqdmUpdate(total=100, desc="Performing broad photo matching...")

        self.Chunk.matchPhotos(cameras=self.Chunk.cameras, downscale=1, generic_preselection=True,
                               reference_preselection=False,
                               reset_matches=True, keep_keypoints=True,
                               keypoint_limit=40000, tiepoint_limit=4000,
                               progress=broad_match_photos_progress_bar.update_to)

    def on_enter_broad_align_cameras(self):
        broad_photos_alignment_progress_bar = TqdmUpdate(total=100, desc="Performing broad camera alignment.")

        self.Chunk.alignCameras(cameras=self.Chunk.cameras, adaptive_fitting=True, reset_alignment=True,
                                progress=broad_photos_alignment_progress_bar.update_to)

    def on_enter_iterative_align(self):
        logger.info("Iteratively aligning cameras.")

        iterative_align_cameras(self.Chunk, self.set_size)

        self.alignment_count += 1

    def on_second_iterative_align(self):
        logger.info("Iteratively aligning cameras (2nd run).")

        iterative_align_cameras(self.Chunk, self.set_size)

        self.alignment_count += 1

    def on_enter_optimize_alignment(self):
        logger.info("Iteratively optimizing alignment.")

        optimize_alignment(self.Chunk)

    def on_enter_filter_uncertainty(self):
        logger.info("Filtering Uncertainty.")

        filter_reconstruction_uncertainty(self.Chunk)

    def on_enter_filter_accuracy(self):
        filter_projection_accuracy(self.Chunk)

    def on_enter_build_rough_model(self):
        build_rough_model_progress_bar = TqdmUpdate(total=100, desc="Building Rough Model")
        self.Chunk.buildModel(surface_type=Metashape.SurfaceType.Arbitrary,
                              source_data=Metashape.DataSource.TiePointsData,
                              progress=build_rough_model_progress_bar.update_to,
                              subdivide_task=True)

    def on_enter_reduce_overlap(self):
        reduce_overlap_progress_bar = TqdmUpdate(total=100, desc="Reducing camera overlap")
        self.Chunk.reduceOverlap(overlap=6, progress=reduce_overlap_progress_bar.update_to)

    def on_enter_generate_masks(self):
        self.Chunk.generateMasks(masking_mode=Metashape.MaskingModeModel,
                                 mask_operation=Metashape.MaskOperationIntersection)

    def on_enter_build_depth_maps(self):
        build_depths_progress_bar = TqdmUpdate(total=100, desc="Building initial depth maps")

        # build depth map in ultra quality
        self.Chunk.buildDepthMaps(downscale=1, filter_mode=Metashape.MildFiltering, max_neighbors=100,
                                  cameras=self.Chunk.cameras, progress=build_depths_progress_bar.update_to)

    def on_enter_build_point_cloud(self):
        build_point_cloud_progress_bar = TqdmUpdate(total=100, desc="Building point cloud")

        self.Chunk.buildPointCloud(source_data=Metashape.DataSource.DepthMapsData, point_confidence=True,
                                   keep_depth=True, progress=build_point_cloud_progress_bar.update_to)

    def on_enter_export_point_cloud(self):
        export_point_cloud_progress_bar = TqdmUpdate(total=100, desc="Exporting Point Cloud")

        self.Chunk.exportPointCloud(str((self.export_path / "pointcloud.ply").resolve()),
                                    source_data=Metashape.DataSource.PointCloudData,
                                    progress=export_point_cloud_progress_bar.update_to)

    def on_exit_export_point_cloud(self):
        shutil.rmtree(self.filename)

    def run(self):
        while self.current_state not in self.final_states:
            self.send("cycle")


def create_or_load_metashape_project(data: Path) -> Tuple[Metashape.Document, Metashape.Chunk, bool]:
    project_path = data / "metashape_project.psx"
    project_path = project_path.resolve()

    # Create a Metashape document
    doc = Metashape.Document()
    loaded = False

    doc.read_only = False

    # If the project file already exists, load it
    if project_path.exists():
        logger.info("Loading existing project")
        doc.open(str(project_path), read_only=False, ignore_lock=True)
        # Assuming that we'd want to work with the first chunk in the project
        chunk = doc.chunk if len(doc.chunks) > 0 else doc.addChunk()
        loaded = True
    else:
        # Else create and save a new project
        logger.info("Creating new project")
        doc.save(path=str(project_path))
        doc.read_only = False
        chunk = doc.addChunk()

    return doc, chunk, loaded


def add_frames_to_chunk(chunk: Metashape.Chunk, frames: List[Path]):
    logger.info(f"Adding {len(frames)} frames to chunk")
    chunk.addPhotos([str(path) for path in frames])


def load_masks(chunk: Metashape.Chunk, mask_path: Path):
    mask_file_path = str(mask_path.resolve()) + "/{filename}_mask.png"

    load_masks_loading_bar = TqdmUpdate(total=100, desc="Loading Masks")

    try:
        chunk.generateMasks(path=mask_file_path, masking_mode=Metashape.MaskingModeFile,
                            progress=load_masks_loading_bar.update_to)

    except Exception as e:
        logger.error(f"An error occurred while generating masks: {e}. Continuing...")


def handle_missing_mask(camera_label: str):
    logger.error(f"No mask file found for camera: {camera_label}")


def handle_error(e: Exception):
    logger.error(f"An error occurred: {e}")


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

    logger.info(f"Total sets for matching: {total_sets}")

    with tqdm(total=total_sets, desc="Matching", dynamic_ncols=True) as pbar:
        for i in range(0, number_of_cameras, set_size):
            match_list = list()

            start_index = max(0, i - overlap)
            end_index = min(number_of_cameras, i + set_size + overlap)

            for j, camera in enumerate(chunk.cameras[start_index:end_index]):
                match_list.append(camera)

            matching_bar = TqdmUpdate(total=100, desc=f"Matching images, iteration: {i} ")

            logger.info(f"Matching photos {start_index} to {end_index}")
            try:
                chunk.matchPhotos(cameras=match_list, downscale=1, generic_preselection=False,
                                  reference_preselection=False, keep_keypoints=True, keypoint_limit=60000,
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
        for camera in tqdm(chunk.cameras, desc=f"Iteratively aligning... Iteration {iteration + 1}",
                           dynamic_ncols=True):
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
            logger.info(f"Stopped realignment after {max_iterations} iterations.")
        else:
            logger.info(f"All cameras realigned after {iteration} iterations.")

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

    logger.info(f"Removing {len(cameras_to_remove)} low quality cameras")
    chunk.remove(cameras_to_remove)


def get_reprojection_error(i, filter):
    return filter.values[i]


def optimize_alignment(chunk: Metashape.Chunk, upper_percentile: float = 90, lower_percentile: float = 10,
                       delta_ratio: float = 0.1, total_allowable_error: float = 0.5, iterations: int = 4) -> None:
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

    logger.info("Optimizing alignment")

    points = chunk.tie_points.points

    if chunk.tie_points.points is None:
        logger.info("No tie points found")
        return

    logger.debug("Got points...")

    tie_point_filter_error = Metashape.TiePoints.Filter()

    tie_point_filter_error.init(chunk, criterion=Metashape.TiePoints.Filter.ReprojectionError)

    logger.info("Initialized filter...")

    reprojection_errors = [error for i, error in enumerate(tie_point_filter_error.values) if points[i].valid]

    logger.info("Calculated reprojection error...")

    threshold = np.percentile(reprojection_errors, upper_percentile)
    final_threshold = np.percentile(reprojection_errors, lower_percentile)

    decrement = (threshold - final_threshold) / iterations

    old_mean_error = np.mean(reprojection_errors)

    error_delta = old_mean_error * delta_ratio

    logger.info(f"threshold {threshold}, final threshold {final_threshold}, decrement {decrement},"
                f"old mean error {old_mean_error}, error delta {error_delta}")

    for _ in tqdm(range(iterations), desc="Optimizing alignment"):
        tie_point_filter_error.selectPoints(threshold)
        chunk.tie_points.removeSelectedPoints()

        threshold -= decrement
        optimize_cameras(chunk)

        reprojection_errors = [error for i, error in enumerate(tie_point_filter_error.values) if points[i].valid]
        new_mean_error = np.mean(reprojection_errors)

        error = abs(new_mean_error - old_mean_error)
        if error < error_delta or new_mean_error < total_allowable_error:
            break

        logger.info(f"Iteration finished; mean error {new_mean_error}")
        old_mean_error = new_mean_error

    logger.info("Finished optimizing alignment...")


def filter(chunk: Chunk, criterion, threshold: float):
    tie_point_filter = Metashape.TiePoints.Filter()
    tie_point_filter.init(chunk, criterion=criterion)

    tie_point_filter.selectPoints(threshold)
    chunk.tie_points.removeSelectedPoints()
    chunk.optimizeCameras(fit_corrections=False)


# 75 based on trial and error; supported by https://doi.org/10.1007/s00468-019-01866-x
def filter_reconstruction_uncertainty(chunk: Chunk, threshold: float = 75):
    logger.info("Filtering for reconstruction uncertainty")
    filter(chunk, Metashape.TiePoints.Filter.ReconstructionUncertainty, threshold)


def filter_projection_accuracy(chunk: Chunk, threshold: float = 10):
    logger.info("Filtering for projection accuracy...")
    filter(chunk, Metashape.TiePoints.Filter.ProjectionAccuracy, threshold)


def process_frames(data: Path, frames: Path, export: Path, mask_path: Union[Path, None] = None):
    state_machine = MetashapeMachine(data, frames, mask_path, export)

    state_machine.run()


if __name__ == "__main__":
    sm = MetashapeMachine(Path("data"), Path("frames"))
    img_path = str(Path("machine.png").resolve())
    sm._graph().write_png(img_path)
