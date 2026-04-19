"""Parallel batch hashing using multiprocessing.Pool.

A pool of worker processes, each with its own FINDHasher instance (created
via a pool initializer), processing images via pool.map(). This avoids
the GIL limitation that makes threading ineffective for CPU-bound work.

The workload is embarrassingly parallel — each image is hashed independently
with no shared state between images.
"""

import multiprocessing
import os
from .hasher import FINDHasher

# Module-level variable for the worker's hasher instance.
# Each worker process gets its own via _init_worker().
_worker_hasher = None



def _init_worker(use_scipy_dct, fast_mode):
    """Pool initializer: create one FINDHasher per worker process.

    The FINDHasher constructor precomputes a 16x64 DCT coefficient matrix.
    By doing this once per process (via the initializer) rather than once
    per image, we avoid redundant trigonometric computation across the
    hundreds or thousands of images each worker will process.
    """
    global _worker_hasher
    _worker_hasher = FINDHasher(
        use_scipy_dct=use_scipy_dct, fast_mode=fast_mode
    )


def _hash_single(path):
    """Hash one image file. Must be a module-level function for pickling."""
    return _worker_hasher.fromFile(path)


def batch_hash(image_paths, num_workers=None, use_scipy_dct=False,
               fast_mode=False):
    """Hash multiple images in parallel using multiprocessing.Pool.

    Args:
        image_paths: List of file paths to images.
        num_workers: Number of worker processes.
                     Defaults to os.cpu_count().
        use_scipy_dct: Passed to FINDHasher in each worker.
        fast_mode: Passed to FINDHasher in each worker.

    Returns:
        List of ImageHash objects, same order as image_paths.
    """
    if num_workers is None:
        num_workers = os.cpu_count()

    with multiprocessing.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(use_scipy_dct, fast_mode),
    ) as pool:
        results = pool.map(_hash_single, image_paths)

    return results
