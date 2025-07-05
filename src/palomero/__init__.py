from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("palomero")
except PackageNotFoundError:
    # package is not installed
    pass