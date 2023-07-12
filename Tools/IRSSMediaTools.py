""""
Python Bindings for IRSSMediaTools.
"""
from pathlib import Path
import subprocess

from Tools import downloadable_tool


class MediaTools:
    PLATFORM_DATA = {
        'Windows': {
            'extension': '.exe',
            'url': "https://github.com/lukasdragon/MediaTools/releases/download/latest/win-x64.zip"
        },
        'Darwin': {
            'url': "https://github.com/lukasdragon/MediaTools/releases/download/latest/osx-x64.zip"
        },
        'Linux': {
            'url': "https://github.com/lukasdragon/MediaTools/releases/download/latest/linux-x64.zip"
        },
    }

    def __init__(self):
        self.tool = downloadable_tool.DownloadableTool(
            tool_name="IRSSMediaTools",
            platform_data=self.PLATFORM_DATA)
        self.tool.setup()

    def extract_frames(self, video_path: Path, output_path: Path, drop: float = 0.5) -> None:
        """
        Extract frames from the video at the given path.

        Args:
            video_path: The path to the video file.
            output_path: The path to save the extracted frames.
            drop: The drop rate for frame extraction.

        Returns:
            None
        """
        print(f"Extracting frames from video at path: {video_path}")

        command = f'extract --input {video_path.resolve()} --output {output_path.resolve()} --drop {drop}'

        try:
            exit_code = self.tool.run_command(command)
            if exit_code != 0:
                raise subprocess.CalledProcessError(exit_code, command)
        except Exception as e:
            print(f"Exception occurred while extracting frames from video at path: {video_path}. Details: {e}")
            raise e
