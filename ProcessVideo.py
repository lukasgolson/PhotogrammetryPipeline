import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

DATA_PATH = Path("data/video")
TEMPORARY_PATH = Path("tmp")


def extract_frames(video_path: Path, temp_path: Path, drop: float = 0.5) -> None:
    """
    Extract frames from the video at the given path.
    """
    print(f"Extracting frames from video at path: {video_path}")

    command = f"IRSSMediaTools extract --input {video_path} --output {temp_path} --drop {drop}"

    try:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Print stdout and stderr in real time
        while True:
            output = process.stdout.readline()
            if process.poll() is not None and output == b'':
                break
            if output:
                print(output.decode().strip())
        _, stderr = process.communicate()
        if stderr:
            print(stderr.decode().strip(), file=sys.stderr)
    except Exception as e:
        print(f"Exception occurred while extracting frames from video at path: {video_path}. Details: {e}")


def get_all_files(dir_path: Path) -> List[Path]:
    """
    Retrieve all video files in the specified directory.
    """
    return [entry for entry in dir_path.iterdir() if entry.is_file()]

def process_videos(data_path: Path, temp_path: Path) -> None:
    """
    Process all video files in the specified directory.
    """
    # Delete the temporary directory if it exists
    if temp_path.exists():
        shutil.rmtree(temp_path)

    video_files = get_all_files(data_path)
    for video_file in video_files:
        print(f"Processing: {video_file}")
        extract_frames(video_file, temp_path)
        print(f"Done processing: {video_file}")


if __name__ == '__main__':
    process_videos(DATA_PATH, TEMPORARY_PATH)
