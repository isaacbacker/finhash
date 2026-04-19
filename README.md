# finhash

A Python library for FINd perceptual image hashing. Computes 256-bit hashes that capture the visual structure of images — similar images produce hashes with small Hamming distance, enabling efficient image matching and clustering at scale.

## Installation

```bash
pip install .
```

Or with development dependencies (pytest, profiling tools):

```bash
pip install ".[dev]"
```

## Quick Start

### Python API

```python
from finhash import FINDHasher

hasher = FINDHasher()

# Hash a single image
h = hasher.fromFile("image.jpg")
print(h)  # 256-bit hex string

# Compare two images
h1 = hasher.fromFile("image1.jpg")
h2 = hasher.fromFile("image2.jpg")
distance = h1 - h2  # Hamming distance (0 = identical, 256 = maximally different)
```

### Docker API

Build and run:

```bash
docker build -t finhash .
docker run -p 8945:8945 finhash
```

Compare two images:

```bash
curl -X POST "http://127.0.0.1:8945/compare" \
  -F "image1=@image1.jpg" \
  -F "image2=@image2.jpg"
```

Response:

```json
{"image1_hash": "393b246d...", "image2_hash": "18ab6c6f...", "distance": 40}
```

Interactive API documentation is available at `http://127.0.0.1:8945/docs`.

## Configuration

`FINDHasher` accepts optional parameters:

| Parameter | Default | Description |
|---|---|---|
| `fast_mode` | `False` | Skip box filter, resize directly to 64×64 via LANCZOS. ~3× faster with equivalent accuracy. |
| `use_scipy_dct` | `False` | Use scipy's FFT-based DCT instead of numpy matrix multiplication. Slightly faster. |
| `cache_size` | `0` | LRU cache size for `fromFile()`. Set >0 to cache hashes of previously seen file paths. |

```python
# Fast mode for high-throughput batch processing
hasher = FINDHasher(fast_mode=True, cache_size=1024)
```

### Batch processing

For hashing many images in parallel:

```python
from finhash import batch_hash

hashes = batch_hash(list_of_paths, num_workers=8, fast_mode=True)
```

Uses `multiprocessing.Pool` to distribute work across CPU cores.

## How It Works

FINd computes a 256-bit perceptual hash through the following pipeline:

1. **Resize** to max 512×512 (preserving aspect ratio)
2. **RGB → Luminance** using BT.601 coefficients
3. **Box filter** — moving-average blur to smooth noise (skipped in `fast_mode`)
4. **Decimate** to 64×64
5. **Partial DCT** — 64×64 → 16×16 discrete cosine transform (low-frequency AC coefficients only, DC excluded)
6. **Median threshold** — each DCT coefficient above the median → 1, below → 0

The resulting 256-bit hash captures the image's broad structural features. Similar images (crops, rescales, minor edits) produce hashes with small Hamming distance.

## Running Tests

```bash
pip install ".[dev]"
python -m unittest discover -s tests
```

Without the meme image dataset, 4 structural tests run and 17 are skipped. To enable the full test suite, place the meme images from the [summative2026 repository](https://github.com/oii-sds-inpractice/summative2026) at `summative2026/meme_images/` relative to the project root:

```
<project root>/
    summative2026/
        meme_images/
            0000_12268686.jpg
            ...
        FINd.py          (optional — needed for one cross-check test)
    src/
    tests/
```

## Project Structure

```
src/finhash/
    __init__.py      — Public API: FINDHasher, MatrixUtil, batch_hash
    hasher.py        — FINDHasher class (numpy/scipy-vectorised pipeline)
    parallel.py      — batch_hash() for multiprocessing
    matrix.py        — MatrixUtil (retained for API compatibility)
    api.py           — FastAPI REST endpoint
tests/
    test_hasher.py   — Hasher correctness tests (12 tests)
    test_parallel.py — Multiprocessing tests (5 tests)
    test_api.py      — API endpoint tests (4 tests)
Dockerfile           — Container image, port 8945
pyproject.toml       — Package metadata and dependencies
```

## Dependencies

- **Runtime**: Pillow, numpy, scipy, imagehash, fastapi, uvicorn, python-multipart
- **Dev**: pytest, httpx, line-profiler, memory-profiler, matplotlib

## License

MIT
