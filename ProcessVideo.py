# Built-in libraries
import shlex
import subprocess
from pathlib import Path


# Function definitions
def run_command(cmd: str) -> int:
    """
    Run a command in a subprocess.

    Args:
        cmd: The command to run as a string.

    Returns:
        The exit code of the subprocess.
    """
    with subprocess.Popen(shlex.split(cmd, posix=False), stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1,
                          universal_newlines=True) as p:
        while True:
            line = p.stdout.readline()
            if not line:
                break
            print(line)

        exit_code = p.poll()
    return exit_code


def extract_frames(video_path: Path, temp_path: Path, drop: float = 0.5) -> None:
    """
    Extract frames from the video at the given path.

    Args:
        video_path: The path to the video file.
        temp_path: The path to save the extracted frames.
        drop: The drop rate for frame extraction.

    Returns:
        None
    """
    print(f"Extracting frames from video at path: {video_path}")

    command = f'IRSSMediaTools extract --input {video_path.resolve()} --output {temp_path.resolve()} --drop {drop}'

    try:
        run_command(command)
    except Exception as e:
        print(f"Exception occurred while extracting frames from video at path: {video_path}. Details: {e}")
        raise e
