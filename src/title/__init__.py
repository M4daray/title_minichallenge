from typing import Final
import os
import pathlib

__version__ = '0.0.1'

package_path: Final[str]  = pathlib.Path().resolve().parent.resolve()

PATH_DATA: Final[str] = os.path.join(package_path, 'data')
PATH_RAW: Final[str] = os.path.join(PATH_DATA, 'raw')
PATH_INTERMEDIATE: Final[str] = os.path.join(PATH_DATA, 'intermediate')
PATH_PRIMARY: Final[str] = os.path.join(PATH_DATA, 'primary')
PATH_REPORT: Final[str] = os.path.join(PATH_DATA, 'report')