# Test functions
import sys
sys.path.insert(0, "src")  # prepend src to sys.path

import os
import unittest
from fastapi.testclient import TestClient
from finhash.api import app

MEME_DIR = os.path.join(
    os.path.dirname(__file__), "..", "summative2026", "meme_images"
)
MEME_IMAGES_AVAILABLE = os.path.isdir(MEME_DIR)


class TestCompareEndpoint(unittest.TestCase):
    """Test the POST /compare API endpoint.

    Uses FastAPI's TestClient to make requests without starting a real server.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    @unittest.skipUnless(MEME_IMAGES_AVAILABLE, "meme dataset not present")
    def test_compare_returns_correct_format(self):
        """Response contains image1_hash, image2_hash, and distance."""
        img1_path = os.path.join(MEME_DIR, "0000_12268686.jpg")
        img2_path = os.path.join(MEME_DIR, "0000_12270286.jpg")
        with open(img1_path, "rb") as f1, open(img2_path, "rb") as f2:
            response = self.client.post(
                "/compare",
                files={"image1": f1, "image2": f2},
            )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("image1_hash", data)
        self.assertIn("image2_hash", data)
        self.assertIn("distance", data)

    @unittest.skipUnless(MEME_IMAGES_AVAILABLE, "meme dataset not present")
    def test_compare_matches_reference_example(self):
        """Response matches the expected reference output.

        Reference curl example expected output:
          image1_hash: 393b246d65a694dc...
          image2_hash: 18ab6c6f4cae949c...
          distance: 40
        """
        img1_path = os.path.join(MEME_DIR, "0000_12268686.jpg")
        img2_path = os.path.join(MEME_DIR, "0000_12270286.jpg")
        with open(img1_path, "rb") as f1, open(img2_path, "rb") as f2:
            response = self.client.post(
                "/compare",
                files={"image1": f1, "image2": f2},
            )
        data = response.json()
        self.assertEqual(
            data["image1_hash"],
            "393b246d65a694dc5386279b8e7394f04c9da697877b18a31995ab9893235b65",
        )
        self.assertEqual(
            data["image2_hash"],
            "18ab6c6f4cae949c591e27998c3394f8588da4d7a74b18a332d6a3d89363db65",
        )
        self.assertEqual(data["distance"], 40)

    @unittest.skipUnless(MEME_IMAGES_AVAILABLE, "meme dataset not present")
    def test_compare_same_image_distance_zero(self):
        """Comparing an image to itself returns distance 0."""
        img_path = os.path.join(MEME_DIR, "0000_12268686.jpg")
        with open(img_path, "rb") as f1, open(img_path, "rb") as f2:
            response = self.client.post(
                "/compare",
                files={"image1": f1, "image2": f2},
            )
        data = response.json()
        self.assertEqual(data["distance"], 0)

    @unittest.skipUnless(MEME_IMAGES_AVAILABLE, "meme dataset not present")
    def test_compare_missing_file_returns_422(self):
        """Omitting one image returns a validation error."""
        img_path = os.path.join(MEME_DIR, "0000_12268686.jpg")
        with open(img_path, "rb") as f1:
            response = self.client.post(
                "/compare",
                files={"image1": f1},
            )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
