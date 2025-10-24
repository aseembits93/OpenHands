import os
from pathlib import Path

__package_name__ = 'openhands_ai'


def get_version():
    try:
        # Calculate root_dir only once
        _abspath = os.path.abspath(__file__)
        root_dir = os.path.dirname(os.path.dirname(_abspath))
        candidate_paths = [
            Path(root_dir) / 'pyproject.toml',
            Path(root_dir) / 'openhands' / 'pyproject.toml',
        ]
        for file_path in candidate_paths:
            if file_path.is_file():
                with open(file_path, 'r') as f:
                    version_line = next(
                        (line for line in f if line.lstrip().startswith('version =')), None
                    )
                    if version_line:
                        v = version_line.split('=', 1)[1]
                        return v.strip().strip('"').strip("'")
    except FileNotFoundError:
        pass

    try:
        from importlib.metadata import version

        from openhands.version import __package_name__  # resolves to string
        return version(__package_name__)
    except (ImportError, ModuleNotFoundError, AttributeError, NameError, Exception):
        pass

    try:
        from pkg_resources import DistributionNotFound  # type: ignore
        from pkg_resources import get_distribution

        from openhands.version import __package_name__
        return get_distribution(__package_name__).version
    except (ImportError, DistributionNotFound, ModuleNotFoundError, AttributeError, NameError, Exception):
        pass

    return 'unknown'


try:
    __version__ = get_version()
except Exception:
    __version__ = 'unknown'
