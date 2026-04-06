"""
Environment validation utility.

Ensures that scripts are run with an officially supported Python
runtime to maintain reproducibility and avoid subtle library behavior
shifts across versions.
"""

import logging
import sys

logger = logging.getLogger(__name__)

SUPPORTED_MAJOR = 3
SUPPORTED_MINOR_MIN = 11
SUPPORTED_MINOR_MAX = 13


def require_supported_runtime() -> None:
    """
    Verify that the current Python interpreter is within the supported
    version range: Python 3.11–3.13.

    Logs a warning (but does not raise) if the version is outside the
    tested range.
    """
    major, minor = sys.version_info.major, sys.version_info.minor

    if major != SUPPORTED_MAJOR or not (SUPPORTED_MINOR_MIN <= minor <= SUPPORTED_MINOR_MAX):
        msg = (
            f"WARNING: CLEVER is tested with Python "
            f"{SUPPORTED_MAJOR}.{SUPPORTED_MINOR_MIN}–"
            f"{SUPPORTED_MAJOR}.{SUPPORTED_MINOR_MAX}. "
            f"Found: Python {major}.{minor}.{sys.version_info.micro}. "
            f"Results may differ slightly across versions."
        )
        logger.warning(msg)

    logger.debug(f"Environment check passed: Python {major}.{minor}.{sys.version_info.micro}")


def pin_numpy_threads() -> None:
    """
    Pin common scientific computing libraries to a single thread
    to prevent oversubscription on shared compute nodes.

    If a thread variable is already set (e.g. by a Slurm script),
    its value is preserved.  Otherwise it defaults to ``"1"``.
    """
    import os

    thread_vars = [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]
    for var in thread_vars:
        if var not in os.environ:
            os.environ[var] = "1"

    omp = os.environ.get("OMP_NUM_THREADS", "1")
    logger.info(f"Numpy thread pinning: OMP_NUM_THREADS={omp}")
