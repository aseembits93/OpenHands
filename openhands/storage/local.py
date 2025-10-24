import os
import shutil

from openhands.core.logger import openhands_logger as logger
from openhands.storage.files import FileStore


class LocalFileStore(FileStore):
    root: str

    def __init__(self, root: str):
        if root.startswith('~'):
            root = os.path.expanduser(root)
        self.root = root
        os.makedirs(self.root, exist_ok=True)

    def get_full_path(self, path: str) -> str:
        if path.startswith('/'):
            path = path[1:]
        return os.path.join(self.root, path)

    def write(self, path: str, contents: str | bytes) -> None:
        full_path = self.get_full_path(path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        mode = 'w' if isinstance(contents, str) else 'wb'
        with open(full_path, mode) as f:
            f.write(contents)

    def read(self, path: str) -> str:
        full_path = self.get_full_path(path)
        with open(full_path, 'r') as f:
            return f.read()

    def list(self, path: str) -> list[str]:
        full_path = self.get_full_path(path)
        try:
            entries = os.scandir(full_path)
        except FileNotFoundError:
            # To keep behavior identical if not previously handled, let the exception bubble up
            raise

        result = []
        # Cache the prefix for construction to avoid repeated os.path.join calls and repeated get_full_path
        prefix = path.rstrip('/')

        for entry in entries:
            entry_name = entry.name
            p = prefix + '/' + entry_name if prefix else entry_name
            if entry.is_dir():
                p += '/'
            result.append(p)

        return result

    def delete(self, path: str) -> None:
        try:
            full_path = self.get_full_path(path)
            if not os.path.exists(full_path):
                logger.debug(f'Local path does not exist: {full_path}')
                return
            if os.path.isfile(full_path):
                os.remove(full_path)
                logger.debug(f'Removed local file: {full_path}')
            elif os.path.isdir(full_path):
                shutil.rmtree(full_path)
                logger.debug(f'Removed local directory: {full_path}')
        except Exception as e:
            logger.error(f'Error clearing local file store: {str(e)}')
