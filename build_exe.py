#!/usr/bin/env python
"""
Build FIPread into a standalone Windows executable.

Default behavior:
    python build_exe.py

The default executable name is FIP.YYYY.MM.DD.exe. Existing executables in
dist are preserved. If the default output file already exists, the new build is
saved with a time suffix instead of overwriting the old executable.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
ENTRY_SCRIPT = PROJECT_ROOT / "run.py"
SRC_DIR = PROJECT_ROOT / "src"
DIST_DIR = PROJECT_ROOT / "dist"
BUILD_DIR = PROJECT_ROOT / "build"
ICON_SOURCE = PROJECT_ROOT / "logo.png"


def default_app_name() -> str:
    return datetime.now().strftime("FIP.%Y.%m.%d")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use PyInstaller to package FIPread into a standalone exe."
    )
    parser.add_argument(
        "--console",
        action="store_true",
        help="Build a console-mode executable for debugging.",
    )
    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="Only remove packaging intermediate files and then exit. dist/*.exe is preserved.",
    )
    parser.add_argument(
        "--skip-clean",
        dest="skip_pre_clean",
        action="store_true",
        help="Do not remove previous intermediate files before packaging.",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep PyInstaller build/spec temporary files after a successful build.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite dist/<name>.exe if it already exists. Default preserves existing exe files.",
    )
    parser.add_argument(
        "--name",
        default=default_app_name(),
        help="Executable base name without .exe. Default: FIP.YYYY.MM.DD",
    )
    parser.add_argument(
        "--distpath",
        default=str(DIST_DIR),
        help=f"Final exe output directory. Default: {DIST_DIR}",
    )
    parser.add_argument(
        "--workpath",
        default=str(BUILD_DIR),
        help=f"Intermediate build directory. Default: {BUILD_DIR}",
    )
    parser.add_argument(
        "--upx-dir",
        default=None,
        help="Optional UPX directory path passed through to PyInstaller.",
    )
    parser.add_argument(
        "--no-icon",
        action="store_true",
        help="Do not convert logo.png into an exe icon.",
    )
    parser.add_argument(
        "--collect-sklearn",
        action="store_true",
        help="Collect all sklearn submodules. Slower and larger; use only if the packaged SVM model reports a missing sklearn module.",
    )
    parser.add_argument(
        "--no-collect-sklearn",
        action="store_false",
        dest="collect_sklearn",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def ensure_project_path(path: Path, label: str) -> Path:
    resolved = path.resolve()
    project_root = PROJECT_ROOT.resolve()
    if resolved == project_root or project_root not in resolved.parents:
        raise RuntimeError(f"Refusing to clean {label} outside the project build area: {resolved}")
    return resolved


def remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
        print(f"[clean] Removed directory: {path}")
        return
    path.unlink()
    print(f"[clean] Removed file: {path}")


def clean_intermediate(work_root: Path) -> None:
    safe_work_root = ensure_project_path(work_root, "workpath")
    remove_path(safe_work_root)


def ensure_entry_script() -> None:
    if not ENTRY_SCRIPT.exists():
        raise FileNotFoundError(f"Entry script not found: {ENTRY_SCRIPT}")
    if not SRC_DIR.exists():
        raise FileNotFoundError(f"Source directory not found: {SRC_DIR}")


def ensure_pyinstaller() -> None:
    try:
        import PyInstaller  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "PyInstaller is not installed. Install it first with:\n"
            "    pip install pyinstaller"
        ) from exc


def add_data_arg(source: Path, destination: str) -> str:
    separator = ";" if os.name == "nt" else ":"
    return f"{source}{separator}{destination}"


def collect_data_files() -> List[str]:
    data_args: List[str] = []
    candidates = [
        (PROJECT_ROOT / "models" / "saved_models", "models/saved_models", True),
        (ICON_SOURCE, ".", False),
    ]
    for source, destination, required in candidates:
        if source.exists():
            data_args.append(add_data_arg(source, destination))
        elif required:
            raise FileNotFoundError(f"Required packaging resource not found: {source}")
        else:
            print(f"[warn] Optional packaging resource not found: {source}")
    return data_args


def prepare_icon_file(work_root: Path, no_icon: bool) -> Path | None:
    if no_icon:
        return None
    if not ICON_SOURCE.exists():
        print(f"[warn] Icon source not found, skip exe icon: {ICON_SOURCE}")
        return None
    try:
        from PIL import Image
    except ImportError:
        print("[warn] Pillow is not installed, skip exe icon conversion.")
        return None

    icon_path = Path(tempfile.gettempdir()) / "fip_build_icon.ico"
    with Image.open(ICON_SOURCE) as image:
        image = image.convert("RGBA")
        image.save(
            icon_path,
            format="ICO",
            sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)],
        )
    print(f"[build] Prepared icon: {icon_path}")
    return icon_path


def build_hidden_imports() -> List[str]:
    return [
        "config",
        "data_access",
        "main_window",
        "models",
        "plotting",
        "processing",
        "PyQt5.QtMultimedia",
        "joblib",
        "matplotlib",
        "matplotlib.cm",
        "matplotlib.colors",
        "nptdms",
        "numpy",
        "pandas",
        "pyqtgraph",
        "scipy",
        "scipy.io.wavfile",
        "scipy.signal",
        "scipy.stats",
        "sklearn",
        "sklearn.pipeline",
        "sklearn.preprocessing",
        "sklearn.preprocessing._data",
        "sklearn.svm",
        "sklearn.svm._classes",
        "sklearn.svm._libsvm",
    ]


def build_excluded_modules() -> List[str]:
    return [
        "PyQt6",
        "PyQt6.QtCore",
        "PyQt6.QtGui",
        "PyQt6.QtWidgets",
        "PySide2",
        "PySide2.QtCore",
        "PySide2.QtGui",
        "PySide2.QtWidgets",
        "PySide6",
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
        "IPython",
        "jedi",
        "notebook",
        "numba",
        "OpenGL",
        "pytest",
        "tkinter",
        "torch",
        "zmq",
    ]


def build_pyinstaller_command(args: argparse.Namespace, work_root: Path) -> List[str]:
    pyinstaller_dist = work_root / "dist"
    pyinstaller_work = work_root / "work"
    pyinstaller_spec = work_root / "spec"
    icon_path = prepare_icon_file(work_root, args.no_icon)

    command: List[str] = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--name",
        args.name,
        "--distpath",
        str(pyinstaller_dist),
        "--workpath",
        str(pyinstaller_work),
        "--specpath",
        str(pyinstaller_spec),
        "--paths",
        str(SRC_DIR),
    ]

    if icon_path is not None:
        command.extend(["--icon", str(icon_path)])
    if not args.console:
        command.append("--windowed")
    if args.upx_dir:
        command.extend(["--upx-dir", str(Path(args.upx_dir).resolve())])

    for data_arg in collect_data_files():
        command.extend(["--add-data", data_arg])
    for hidden_import in build_hidden_imports():
        command.extend(["--hidden-import", hidden_import])
    if args.collect_sklearn:
        command.extend(["--collect-submodules", "sklearn"])
    for excluded_module in build_excluded_modules():
        command.extend(["--exclude-module", excluded_module])

    command.append(str(ENTRY_SCRIPT))
    return command


def run_command(command: Sequence[str]) -> None:
    print("[build] Running command:")
    print(f"        {subprocess.list2cmdline(list(command))}")
    subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)


def unique_output_path(app_name: str, distpath: Path, overwrite: bool) -> Path:
    target = distpath / f"{app_name}.exe"
    if overwrite or not target.exists():
        return target

    timestamp = datetime.now().strftime("%H%M%S")
    candidate = distpath / f"{app_name}.{timestamp}.exe"
    counter = 2
    while candidate.exists():
        candidate = distpath / f"{app_name}.{timestamp}.{counter}.exe"
        counter += 1
    print(f"[warn] Existing exe preserved: {target}")
    print(f"[warn] New exe will be saved as: {candidate.name}")
    return candidate


def move_final_exe(app_name: str, distpath: Path, work_root: Path, overwrite: bool) -> Path:
    built_exe = work_root / "dist" / f"{app_name}.exe"
    if not built_exe.exists():
        raise FileNotFoundError(f"Packaging finished but exe was not found: {built_exe}")

    distpath.mkdir(parents=True, exist_ok=True)
    final_exe = unique_output_path(app_name, distpath, overwrite)
    if overwrite and final_exe.exists():
        final_exe.unlink()
        print(f"[post] Overwrote existing exe: {final_exe}")
    shutil.move(str(built_exe), str(final_exe))
    return final_exe


def print_summary(exe_path: Path, cleaned: bool) -> None:
    size_mb = exe_path.stat().st_size / (1024 * 1024)
    print()
    print("[done] Packaging completed.")
    print(f"[done] EXE path: {exe_path}")
    print(f"[done] EXE size: {size_mb:.2f} MB")
    print(f"[done] Intermediate files cleaned: {'yes' if cleaned else 'no'}")


def main() -> int:
    args = parse_args()
    if os.name != "nt":
        print("[warn] This script is intended for Windows packaging. Current platform is not Windows.")

    work_root = ensure_project_path(Path(args.workpath), "workpath")
    distpath = Path(args.distpath).resolve()

    if not args.skip_pre_clean:
        clean_intermediate(work_root)

    if args.clean_only:
        print("[done] Clean-only mode finished. Existing dist/*.exe files were not removed.")
        return 0

    ensure_entry_script()
    ensure_pyinstaller()

    cleaned = False
    command = build_pyinstaller_command(args, work_root)
    run_command(command)
    exe_path = move_final_exe(args.name, distpath, work_root, args.overwrite)

    if not args.keep_intermediate:
        clean_intermediate(work_root)
        cleaned = True

    print_summary(exe_path, cleaned)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
