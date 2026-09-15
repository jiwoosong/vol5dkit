"""TCSHW tensor geometry and an optional native-slice viewer."""

from .volume import Volume

__version__ = "0.0.0.1"
__all__ = ["Volume", "view", "run", "__version__"]


def view(a, b=None, *, rgb=False, interpolation="nearest", clim=None, block=True):
    """Open a local viewer; install ``vol5dkit[gui]`` for this optional feature.

    Use ``block=False`` with IPython/Jupyter's ``%gui qt6`` integration.
    Inputs are :class:`Volume` instances, with explicit TCSHW axes.
    """
    try:
        from .viewer.window import view as open_viewer
    except ModuleNotFoundError as exc:
        if exc.name in {"PySide6", "vispy"}:
            raise ImportError('The viewer requires: pip install "vol5dkit[gui]"') from exc
        raise
    viewer = open_viewer(a, b, rgb=rgb, interpolation=interpolation, clim=clim, block=False)
    # A blocking call must not retain the original tensors in this stack frame.
    del a, b
    if block:
        run()
    return viewer


def run():
    """Run the Qt event loop after opening one or more nonblocking viewers."""
    try:
        from .viewer.window import run as run_viewers
    except ModuleNotFoundError as exc:
        if exc.name in {"PySide6", "vispy"}:
            raise ImportError('The viewer requires: pip install "vol5dkit[gui]"') from exc
        raise
    return run_viewers()
