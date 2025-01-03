from os import W_OK, PathLike, access
from pathlib import Path


def enforce_filepath(filepath: str | PathLike, extensions: list[str]) -> Path:
    """
    Parameters
    ----------
    filepath : str | PathLike
        Path or filepath to check.
    extensions : [str]
        List of extensions to check for, e.g. [".png", ".pdf"]

    Raises
    ------
    ValueError
        If the provided file path has an invalid extension.
    NotADirectoryError
        If the parent directory of the specified file path does not exist.
        Additionally raised if the provided path is invalid and not a directory.
    PermissionError
        If the specified directory cannot be written to due to insufficient permissions.
    """
    path = Path(filepath)
    filepath = None  # type: ignore
    if path.is_dir():
        filepath = path.joinpath(f"default{extensions[0]}")
    elif len(path.suffix) > 0:
        if path.suffix not in extensions:
            raise ValueError(
                f"Invalid filepath suffix, expected {extensions}, got {path.suffix}"
            )
        if not path.parent.is_dir():
            raise NotADirectoryError(
                "Invalid filepath, parent directory does not exist"
            )
        filepath = path
    else:
        raise NotADirectoryError("Invalid path, directory does not exist")

    if not access(filepath.parent, W_OK):
        raise PermissionError(
            "Cannot write to specified directory, insufficient permissions"
        )
    return filepath
