"""FastAPI REST API for FINd image hashing.

Provides a single endpoint POST /compare that accepts two image files
and returns their FINd hashes and Hamming distance.

Uses FastAPI's @app.post with UploadFile for file uploads and a
Pydantic response model for auto-generated API documentation at /docs.

Usage:
    uvicorn finhash.api:app --host 0.0.0.0 --port 8945
"""

import logging
import time
from io import BytesIO

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from PIL import Image

from .hasher import FINDHasher

# Structured logging for per-request observability.
logger = logging.getLogger("finhash.api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

app = FastAPI(title="FINd Image Hashing API")

# Single hasher instance shared across requests. Standard mode produces
# bit-identical hashes to the original FINd algorithm.
_hasher = FINDHasher()


# Pydantic response model. Provides auto-generated schema documentation
# at /docs, so engineers can see the exact response format without
# reading the source code.
class CompareResponse(BaseModel):
    image1_hash: str
    image2_hash: str
    distance: int


@app.post("/compare", response_model=CompareResponse)
async def compare(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
):
    """Compare two images by their FINd perceptual hashes.

    Accepts two image files as multipart form uploads and returns
    their 256-bit FINd hashes (hex) and Hamming distance.
    """
    try:
        img1 = Image.open(BytesIO(await image1.read()))
        img2 = Image.open(BytesIO(await image2.read()))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    start = time.time()
    h1 = _hasher.fromImage(img1)
    h2 = _hasher.fromImage(img2)
    elapsed = time.time() - start

    distance = int(h1 - h2)
    logger.info(
        "compare: image1=%s image2=%s distance=%d hash_time=%.3fs",
        image1.filename, image2.filename, distance, elapsed,
    )

    return CompareResponse(
        image1_hash=str(h1),
        image2_hash=str(h2),
        distance=distance,
    )
