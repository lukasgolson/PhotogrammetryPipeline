from pathlib import Path
from typing import Optional

from Tools import downloadable_tool

URL = "https://github.com/OpenDroneMap/SkyRemoval/archive/91fe0362b14cb74c7a5383cf3e83d764b749972d.zip"
EXTENSION = ".py"
SUBDIR = "SkyRemoval-91fe0362b14cb74c7a5383cf3e83d764b749972d"
PLATFORMS = ['Windows', 'Darwin', 'Linux']
PLATFORM_DATA = {platform: {'extension': EXTENSION, 'subdir': SUBDIR, 'url': URL} for platform in PLATFORMS}


class SkyRemoval:
    def __init__(self):
        try:

            self.tool = downloadable_tool.DownloadableTool(
                tool_name="SkyRemoval",
                platform_data=PLATFORM_DATA,
                python=True)
            self.tool.setup()
        except Exception as e:
            print(f"Failed to setup the tool: {e}")
            raise

    def remove_sky(self, source: Path, destination: Path, model: Optional[str] = None, ignore_cache: bool = False,
                   in_size_w: Optional[int] = None, in_size_h: Optional[int] = None):
        """
       Remove sky from images using SkyRemoval tool.

       :param self: Instance reference
       :param source: The source image path. This can be a single image or a directory.
       :param destination: The destination directory path where the processed images will be stored.
       :param model: The model path. This can be a URL or a local file path. If not provided, the default model is used.
       :param ignore_cache: If True, cache is ignored when downloading the model. Defaults to False.
       :param in_size_w: The trained model input width. If not provided, the default width is used.
       :param in_size_h: The trained model input height. If not provided, the default height is used.

       :raises Exception: If there is a failure in executing the command to the SkyRemoval tool.
       """

        command = _form_sky_removal_command(source, destination, model, ignore_cache, in_size_w, in_size_h)
        try:
            self.tool.run_command(command)
        except Exception as e:
            print(f"Failed to execute the command: {e}")
            raise


def _form_sky_removal_command(source: Path, destination: Path, model: Optional[str] = None, ignore_cache: bool = False,
                              in_size_w: Optional[int] = None, in_size_h: Optional[int] = None) -> str:
    """Form the command string to pass to the SkyRemoval tool
    :rtype: str
    """
    command = f"{source.resolve()} {destination.resolve()}"
    if model:
        command += f" --model {model}"
    if ignore_cache:
        command += " --ignore_cache"
    if in_size_w:
        command += f" --in_size_w {in_size_w}"
    if in_size_h:
        command += f" --in_size_h {in_size_h}"
    return command


if __name__ == "__main__":
    SkyRemoval()
