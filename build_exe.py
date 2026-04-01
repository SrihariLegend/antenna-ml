"""Build a standalone executable using PyInstaller.

Bundles the Gradio app, ML model, dataset, Balanis module, and CST Linker
into a single dist/antenna-ml/ directory.

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
    data_files: list[tuple[str, str]] = []

    for src, dest in config.PYINSTALLER_BUNDLE_DATA:
        if os.path.exists(src):
            data_files.append((src, dest))
            logger.info("Bundling data: %s -> %s", src, dest)

    # Bundle model artifacts
    for pkl in glob.glob("*.pkl"):
        data_files.append((pkl, "."))
        logger.info("Bundling model artifact: %s", pkl)

    # Bundle CST Linker scripts
    cst_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "CST_Linker")
    if os.path.isdir(cst_dir):
        data_files.append((cst_dir, "cst_linker"))
        logger.info("Bundling CST Linker from: %s", cst_dir)
    else:
        logger.warning("CST_Linker directory not found at %s", cst_dir)

    return data_files


def build() -> None:
    hidden_imports = [
        "sklearn", "sklearn.ensemble", "sklearn.preprocessing", "sklearn.utils",
        "gradio", "pandas", "numpy", "joblib", "webview",
    ]

    data_files = _collect_data_files()

    # Gradio static assets
    try:
        import gradio
        data_files.append((os.path.dirname(gradio.__file__), "gradio"))
    except ImportError:
        logger.warning("Gradio not found")

    args = [
        config.PYINSTALLER_ENTRY,
        "--name", config.PYINSTALLER_NAME,
        "--noconfirm", "--clean",
    ]
    for imp in hidden_imports:
        args.extend(["--hidden-import", imp])
    for src, dest in data_files:
        args.extend(["--add-data", f"{src}{os.pathsep}{dest}"])

    logger.info("Building %s...", config.PYINSTALLER_NAME)
    PyInstaller.__main__.run(args)
    logger.info("Build complete! Output: dist/%s/", config.PYINSTALLER_NAME)


if __name__ == "__main__":
    build()
