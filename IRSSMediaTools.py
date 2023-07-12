"""
Python Bindings for IRSSMediaTools.
"""
from pathlib import Path

import downloadable_tool

# Mapping for platform-specific data.
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

tool = downloadable_tool.DownloadableTool(
    tool_name="IRSSMediaTools",
    platform_data=PLATFORM_DATA
)

tool.setup()

if __name__ == '__main__':
    tool.run_command('help')


def extract_frames(video_path: Path, output_path: Path, drop: float = 0.5) -> None:
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
        tool.run_command(command)
    except Exception as e:
        print(f"Exception occurred while extracting frames from video at path: {video_path}. Details: {e}")
        raise e




if __name__ == '__main__':
    tool.run_command('help')
