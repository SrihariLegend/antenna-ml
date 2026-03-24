"""Build a standalone executable using PyInstaller in single-directory mode.

Bundles the Gradio app, model artifacts, default dataset, and all
dependencies into ``dist/antenna-ml/``.

Usage:
    python build_exe.py
"""

import glob
import logging
import os

import PyInstaller.__main__

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _collect_data_files() -> list[tuple[str, str]]:
    """Gather data files to bundle, skipping entries that don't exist yet."""
    data_files: list[tuple[str, str]] = []

    for src, dest in config.PYINSTALLER_BUNDLE_DATA:
        if os.path.exists(src):
            data_files.append((src, dest))
            logger.info("Bundling data: %s -> %s", src, dest)
        else:
            logger.warning("Skipping missing data path: %s", src)

    # Bundle all .pkl model files found in the project root
    for pkl in glob.glob("*.pkl"):
        data_files.append((pkl, "."))
        logger.info("Bundling model artifact: %s", pkl)

    return data_files


def _find_gradio_data() -> str | None:
    """Locate the Gradio package data directory for static assets."""
    try:
        import gradio

        gradio_dir = os.path.dirname(gradio.__file__)
        logger.info("Found Gradio package at: %s", gradio_dir)
        return gradio_dir
    except ImportError:
        logger.warning("Gradio not found — static assets will not be bundled")
        return None


def build() -> None:
    """Run PyInstaller with the project configuration."""
    hidden_imports = [
        "sklearn",
        "sklearn.ensemble",
        "sklearn.preprocessing",
        "sklearn.utils",
        "gradio",
        "pandas",
        "numpy",
        "joblib",
    ]

    data_files = _collect_data_files()

    # Gradio static assets
    gradio_dir = _find_gradio_data()
    if gradio_dir:
        data_files.append((gradio_dir, "gradio"))

    # Build PyInstaller arguments
    args = [
        config.PYINSTALLER_ENTRY,
        "--name", config.PYINSTALLER_NAME,
        "--noconfirm",
        "--clean",
        # Single-directory mode (default, but explicit for clarity)
    ]

    for imp in hidden_imports:
        args.extend(["--hidden-import", imp])

    for src, dest in data_files:
        args.extend(["--add-data", f"{src}{os.pathsep}{dest}"])

    logger.info("Running PyInstaller with entry point: %s", config.PYINSTALLER_ENTRY)
    logger.info("Output name: %s", config.PYINSTALLER_NAME)
    logger.info("Hidden imports: %s", hidden_imports)
    logger.info("Data files: %d entries", len(data_files))

    PyInstaller.__main__.run(args)

    logger.info("Build complete! Output: dist/%s/", config.PYINSTALLER_NAME)


if __name__ == "__main__":
    build()
