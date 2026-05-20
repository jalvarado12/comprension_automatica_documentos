from __future__ import annotations

import os
import sys
from pathlib import Path


def configure_threading_env() -> None:
    """
    Evita conflictos de threading entre OpenBLAS, MKL y OpenMP.
    Debe llamarse antes de importar numpy/torch.
    Extraído del NB1 donde estaba suelto antes de los imports.
    """
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def add_windows_dll_dirs() -> None:
    """
    Registra directorios de DLL para evitar errores de carga de torch en Windows.
    Solo tiene efecto en Windows; en Linux/Mac es no-op.
    """
    if sys.platform != "win32":
        return

    venv_root = Path(sys.executable).resolve().parents[1]
    candidates = [
        venv_root / "Scripts",
        venv_root / "Lib" / "site-packages" / "torch" / "lib",
    ]
    for dll_dir in candidates:
        if dll_dir.exists():
            try:
                os.add_dll_directory(str(dll_dir))
            except Exception:
                pass
