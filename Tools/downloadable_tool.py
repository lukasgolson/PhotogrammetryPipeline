"""
A framework for managing and executing downloadable tools.
"""

import platform
import shlex
import subprocess
import zipfile
from pathlib import Path
from typing import Dict

import requests
from tqdm import tqdm


class DownloadableTool:
    BASE_DIRECTORY = Path("../third-party")
    CHUNK_SIZE = 1024  # define magic constant

    def __init__(self, tool_name: str, platform_data: Dict[str, dict], python: bool = False):
        self.tool_name = tool_name
        self.platform_data = platform_data
        self.tool_directory = self.BASE_DIRECTORY / self.tool_name
        self.python = python  # add python flag as an object field

    def get_platform_data(self) -> dict:
        """Retrieves platform-specific data from the dictionary."""
        system = platform.system()
        if system not in self.platform_data:
            raise ValueError(f"Unsupported operating system: {system}")
        return self.platform_data[system]

    def calculate_path(self) -> Path:
        """
        Calculates the path of the tool based on the operating system.
        """
        platform_data = self.get_platform_data()
        subdir = platform_data.get('subdir', "")
        extension = platform_data.get('extension', "")

        if subdir:
            path = self.tool_directory / subdir / f'{self.tool_name}{extension}'
        else:
            path = self.tool_directory / f'{self.tool_name}{extension}'

        print(f"Generated path: {path}")
        return path

    def setup(self) -> None:
        """
        Sets up the tool by downloading and extracting it.
        """
        if self.calculate_path().exists():
            print(f"{self.tool_name} already installed.")
            return

        self.tool_directory.mkdir(parents=True, exist_ok=True)
        print(f"Installing {self.tool_name}...")
        url = self.get_platform_data()['url']
        self._download_and_extract_zip(url)
        print(f"{self.tool_name} installed.")

    def _download_and_extract_zip(self, url: str) -> None:
        """
        Downloads a zip file from the given URL and extracts it.
        """
        local_path = self.tool_directory / "download.zip"

        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"Downloading {url}")
        with open(local_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=self.CHUNK_SIZE):
                if chunk:
                    file.write(chunk)
                    progress_bar.update(len(chunk))
        progress_bar.close()

        print(f"Downloaded {url} to {local_path}")

        with zipfile.ZipFile(local_path, 'r') as zip_ref:
            zip_ref.extractall(self.tool_directory)
        print(f"Extracted all files to {self.tool_directory}")

        local_path.unlink()

    def run_command(self, cmd: str) -> int:
        """
        Run a command in a subprocess.

        Args:
            cmd: The command to run as a string.

        Returns:
            The exit code of the subprocess.
        """
        if self.python:  # use python flag as an object field
            cmd = f'python {cmd}'
        else:
            cmd = f'{self.calculate_path().resolve()} {cmd}'
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