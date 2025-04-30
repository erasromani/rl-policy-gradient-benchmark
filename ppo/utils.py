from datetime import datetime
from os import PathLike
from pathlib import Path


def mkdir_datetime(dir_path: PathLike) -> Path:
    """Create directory with current datetime.

    Args:
        dir_path (PathLike[str]): directory path

    Returns:
        Path: directory path with current datetime
    """

    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d")
    time_string = now.strftime("%H-%M-%S")
    datetime_dir = Path(dir_path) / f"{date_string}/{time_string}"
    datetime_dir.mkdir(parents=True, exist_ok=False)
    return datetime_dir
