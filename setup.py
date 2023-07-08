import subprocess
import sys

from helpers import get_all_files


def install_and_import(package, path_to_whl=None):
    try:
        __import__(package)
        print(f"{package} is already installed.")
    except ImportError:
        print(f"{package} not found. Installing...")
        if path_to_whl is None:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        else:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', path_to_whl])
        print(f"{package} has been installed.")
    finally:
        globals()[package] = __import__(package)


def install_requirements():
    print("Installing packages from requirements.txt...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
    print("All packages from requirements.txt have been installed.")


def main():
    install_requirements()

    whls = get_all_files('bin', '*.whl')
    for whl in whls:
        package_name = whl.stem.split('-')[0]  # Get package name from .whl file
        install_and_import(package_name, f'bin/{whl.name}')


if __name__ == "__main__":
    main()
