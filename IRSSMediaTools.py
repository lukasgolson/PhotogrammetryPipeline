"""
Python Bindings for IRSSMediaTools.
"""

import shlex
import subprocess

import os
import platform
import zipfile
from pathlib import Path

import requests

# Define constant for the base directory.
BASE_DIRECTORY = Path("third-party") / Path("IRSSMediaTools")

# Mapping for platform-specific data.
PLATFORM_DATA = {
    'Windows': {
        'folder': 'win-x64',
        'extension': '.exe',
        'url': "https://github.com/lukasdragon/MediaTools/releases/download/latest/win-x64.zip"
    },
    'Darwin': {
        'folder': 'osx-x64',
        'url': "https://github.com/lukasdragon/MediaTools/releases/download/latest/osx-x64.zip"
    },
    'Linux': {
        'folder': 'linux-x64',
        'url': "https://github.com/lukasdragon/MediaTools/releases/download/latest/linux-x64.zip"
    },
}


def get_platform_data():
    """Retrieves platform-specific data from the dictionary."""

    system = platform.system()
    if system not in PLATFORM_DATA:
        raise ValueError(f"Unsupported operating system: {system}")
    return PLATFORM_DATA[system]


def calculate_path() -> Path:
    """
    Calculates the path of the media tools based on the operating system.
    """
    platform_data = get_platform_data()
    folder = platform_data.get('folder', "")
    extension = platform_data.get('extension', "")

    # Formulate the path using the folder and extension information.
    path = os.path.join(BASE_DIRECTORY, folder, f'IRSSMediaTools{extension}')
    print(f"Generated path: {path}")
    return Path(path)


def setup_irss_media_tools() -> None:
    """
    Sets up the media tools by downloading and extracting them.
    """

    if Path(calculate_path()).exists():
        print("IRSSMediaTools already installed.")
        return

    # Remove the directory if it exists.
    os.makedirs(BASE_DIRECTORY, exist_ok=True)
    print("Installing IRSSMediaTools...")
    url = get_platform_data()['url']
    download_and_extract_zip(url)
    print("IRSSMediaTools installed.")


def download_and_extract_zip(url: str) -> None:
    """
    Downloads a zip file from the given URL and extracts it.
    """
    # Define the path for the downloaded zip file.
    local_path = os.path.join(BASE_DIRECTORY, "download.zip")

    # Make a request to download the file.
    response = requests.get(url, stream=True)

    # Ensure the base directory exists.
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # Write the response content to the file in chunks.
    with open(local_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)

    print(f"Downloaded {url} to {local_path}")

    # Extract the contents of the zip file.
    with zipfile.ZipFile(local_path, 'r') as zip_ref:
        zip_ref.extractall(BASE_DIRECTORY)
    print(f"Extracted all files to {BASE_DIRECTORY}")

    # Clean up by removing the downloaded zip file.
    os.remove(local_path)


def run_command(cmd: str) -> int:
    """
    Run a command in a subprocess.

    Args:
        cmd: The command to run as a string.

    Returns:
        The exit code of the subprocess.
    """

    cmd = f'{calculate_path().resolve()} {cmd}'
    print(f"Running command: {cmd}")

    with subprocess.Popen(shlex.split(cmd, posix=False), stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1,
                          universal_newlines=True) as p:
        while True:
            line = p.stdout.readline()
            if not line:
                break
            print(line.strip())

            # process stderr
        for line in p.stderr:
            print(line.strip())

        exit_code = p.poll()
    return exit_code


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
        run_command(command)
    except Exception as e:
        print(f"Exception occurred while extracting frames from video at path: {video_path}. Details: {e}")
        raise e


"""
Run the setup script when importing to make sure IRSSMediaTools is installed and ready to go.
"""
setup_irss_media_tools()

if __name__ == '__main__':
    run_command('help')
