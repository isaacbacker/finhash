# Test functions
import sys
sys.path.insert(0, "src")  # prepend src to sys.path

import os
import unittest
from finhash import FINDHasher, batch_hash

MEME_DIR = os.path.join(
    os.path.dirname(__file__), "..", "summative2026", "meme_images"
)
MEME_IMAGES_AVAILABLE = os.path.isdir(MEME_DIR)


class TestBatchHash(unittest.TestCase):
    """Verify that multiprocessing batch_hash produces identical results
    to sequential hashing. The parallel implementation must not alter
    hash values — it is purely a throughput optimization."""

    @classmethod
    def setUpClass(cls):
        """Compute sequential reference hashes for the first 20 images."""
        cls.hasher = FINDHasher()
        if MEME_IMAGES_AVAILABLE:
            cls.image_paths = [
                os.path.join(MEME_DIR, f)
                for f in sorted(os.listdir(MEME_DIR))[:20]
            ]
            cls.sequential_hashes = [
                str(cls.hasher.fromFile(p)) for p in cls.image_paths
            ]
        else:
            cls.image_paths = []
            cls.sequential_hashes = []

    @unittest.skipUnless(MEME_IMAGES_AVAILABLE, "meme dataset not present")
    def test_parallel_matches_sequential(self):
        """batch_hash results are bit-identical to sequential hashing."""
        parallel_hashes = [str(h) for h in batch_hash(self.image_paths)]
        for i, path in enumerate(self.image_paths):
            with self.subTest(image=os.path.basename(path)):
                self.assertEqual(parallel_hashes[i], self.sequential_hashes[i])

    @unittest.skipUnless(MEME_IMAGES_AVAILABLE, "meme dataset not present")
    def test_parallel_with_2_workers(self):
        """Correctness holds with only 2 worker processes."""
        parallel_hashes = [
            str(h) for h in batch_hash(self.image_paths, num_workers=2)
        ]
        self.assertEqual(parallel_hashes, self.sequential_hashes)

    @unittest.skipUnless(MEME_IMAGES_AVAILABLE, "meme dataset not present")
    def test_parallel_with_1_worker(self):
        """Single worker degenerates to sequential (sanity check)."""
        parallel_hashes = [
            str(h) for h in batch_hash(self.image_paths, num_workers=1)
        ]
        self.assertEqual(parallel_hashes, self.sequential_hashes)

    @unittest.skipUnless(MEME_IMAGES_AVAILABLE, "meme dataset not present")
    def test_preserves_order(self):
        """Results are returned in the same order as input paths."""
        results = batch_hash(self.image_paths)
        self.assertEqual(len(results), len(self.image_paths))
        # Verify first and last match (order-sensitive check)
        self.assertEqual(str(results[0]), self.sequential_hashes[0])
        self.assertEqual(str(results[-1]), self.sequential_hashes[-1])

    def test_empty_input(self):
        """Empty input returns empty output without error."""
        results = batch_hash([])
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
