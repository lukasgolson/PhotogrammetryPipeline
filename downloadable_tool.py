"""
A framework for managing and executing downloadable tools.
"""

import os
import platform
import shlex
import subprocess
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


class DownloadableTool:
    BASE_DIRECTORY = Path("third-party")

    def __init__(self, tool_name: str, platform_data: dict):
        self.tool_name = tool_name
        self.platform_data = platform_data
        self.tool_directory = self.BASE_DIRECTORY / self.tool_name

    def get_platform_data(self):
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
        extension = platform_data.get('extension', "")
        path = os.path.join(self.tool_directory, f'{self.tool_name}{extension}')
        print(f"Generated path: {path}")
        return Path(path)

    def setup(self) -> None:
        """
        Sets up the tool by downloading and extracting it.
        """
        if Path(self.calculate_path()).exists():
            print(f"{self.tool_name} already installed.")
            return

        os.makedirs(self.tool_directory, exist_ok=True)
        print(f"Installing {self.tool_name}...")
        url = self.get_platform_data()['url']
        self.__download_and_extract_zip__(url)
        print(f"{self.tool_name} installed.")

    def __download_and_extract_zip__(self, url: str) -> None:
        """
        Downloads a zip file from the given URL and extracts it.
        """
        local_path = os.path.join(self.tool_directory, "download.zip")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
        with open(local_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
                    progress_bar.update(len(chunk))
        progress_bar.close(leave=False)

        print(f"Downloaded {url} to {local_path}")

        with zipfile.ZipFile(local_path, 'r') as zip_ref:
            zip_ref.extractall(self.tool_directory)
        print(f"Extracted all files to {self.tool_directory}")

        os.remove(local_path)

    def run_command(self, cmd: str) -> int:
        """
        Run a command in a subprocess.

        Args:
            cmd: The command to run as a string.

        Returns:
            The exit code of the subprocess.
        """
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
