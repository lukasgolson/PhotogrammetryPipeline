import subprocess
import sys

from Tools.irss_media_tools import MediaTools
from helpers import get_all_files
from loguru import logger


def install_and_import(package, path_to_whl=None):
    try:
        __import__(package)
        logger.info(f"{package} is already installed.")
    except ImportError:
        logger.info(f"{package} not found. Installing...")
        if path_to_whl is None:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        else:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', path_to_whl])
        logger.info(f"{package} has been installed.")
    finally:
        globals()[package] = __import__(package)


def install_requirements():
    logger.info("Installing packages from requirements.txt...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
    logger.info("All packages from requirements.txt have been installed.")

def main():
    install_requirements()

    whls = get_all_files('../bin', '*.whl')
    for whl in whls:
        package_name = whl.stem.split('-')[0]  # Get package name from .whl file
        install_and_import(package_name, f'bin/{whl.name}')

    MediaTools().tool.setup()


logger.success("Setup complete.")

if __name__ == "__main__":
    main()
