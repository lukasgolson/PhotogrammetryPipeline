"""
Python Bindings for IRSSMediaTools.
"""

from pathlib import Path

from Tools import downloadable_tool, ModelExecutionEngines

from loguru import logger


class MediaTools:
    PLATFORM_DATA = {
        'Windows': {
            'extension': '.exe',
            'url': "https://github.com/IRSS-UBC/MediaTools/releases/download/latest/win-x64.zip"
        },
        'Darwin': {
            'url': "https://github.com/IRSS-UBC/MediaTools/releases/download/latest/osx-x64.zip"
        },
        'Linux': {
            'url': "https://github.com/IRSS-UBC/MediaTools/releases/download/latest/linux-x64.zip"
        },
    }

    def __init__(self, base_dir: Path = "./third-party"):
        self.tool = downloadable_tool.DownloadableTool(
            tool_name="IRSSMediaTools",
            platform_data=self.PLATFORM_DATA,
            python=False,
            base_dir=base_dir)
        self.tool.setup()

    def extract_frames(self, video_path: Path, output_path: Path, drop: float = 0.5) -> None:
        """
        Extract frames from the video_old at the given path.

        Args:
            video_path: The path to the video_old file.
            output_path: The path to save the extracted frames.
            drop: The drop rate for frame extraction.

        Returns:
            None
        """
        logger.info(f"Extracting frames from video_old at path: {video_path}")

        command = f'extract --input "{video_path.resolve()}" ' \
                  f'--output "{output_path.resolve()}" ' \
                  f'--drop {drop} --format png'

        try:
            exit_code = self.tool.run_command(command)

            if exit_code != 0:
                raise Exception(f"Failed to extract frames from video_old at path: {video_path}")
            else:
                logger.success(f"Extracting frames from video_old at path: {video_path} completed with exit code: {exit_code}")


        except Exception as e:
            logger.error(f"Exception occurred while extracting frames from video_old at path: {video_path}. Details: {e}")
            raise e

    def mask_sky(self, images: Path, output_path: Path,
                 engine: ModelExecutionEngines = ModelExecutionEngines.CPU) -> None:

        logger.info(f"Masking sky in images from path: {images}")

        command = f'mask_sky --input "{images.resolve()}" ' \
                  f'--output "{output_path.resolve()}" ' \
                  f'--engine {engine.value}'

        try:
            exit_code = self.tool.run_command(command)

            if exit_code != 0:
                raise Exception(f"Failed to mask images from path: {images}")
            else:
                logger.success(f"Masking images from path: {images} completed with exit code: {exit_code}")


        except Exception as e:
            logger.error(f"Exception occurred while masking images from path: {images}. Details: {e}")
            raise e
