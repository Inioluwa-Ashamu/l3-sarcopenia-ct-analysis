import math
import unittest

import numpy as np

from sarcopenia_metrics import compute_csa_smra, dice_coef, find_max_area_slice


class MetricsTests(unittest.TestCase):
    def test_compute_csa_smra_uses_pixel_area_and_hu_window(self):
        ct = np.array([[0, 10], [200, -50]], dtype=np.int16)
        mask = np.array([[1, 1], [1, 0]], dtype=np.uint8)

        csa, smra, npx = compute_csa_smra(ct, mask, px_area_cm2=0.25)

        self.assertEqual(csa, 0.75)
        self.assertEqual(npx, 3)
        self.assertEqual(smra, 5.0)

    def test_compute_csa_smra_empty_mask_returns_nan_smra(self):
        ct = np.ones((2, 2), dtype=np.int16)
        mask = np.zeros((2, 2), dtype=np.uint8)

        csa, smra, npx = compute_csa_smra(ct, mask, px_area_cm2=0.5)

        self.assertEqual(csa, 0.0)
        self.assertEqual(npx, 0)
        self.assertTrue(math.isnan(smra))

    def test_dice_coef_identical_masks(self):
        mask = np.array([[1, 0], [1, 0]], dtype=np.uint8)

        self.assertGreater(dice_coef(mask, mask), 0.999)

    def test_find_max_area_slice_selects_largest_l3_area(self):
        mask = np.zeros((3, 4, 4), dtype=np.uint8)
        mask[0, :1, :1] = 1
        mask[1, :2, :2] = 1
        mask[2, :1, :2] = 1

        self.assertEqual(find_max_area_slice(mask), 1)


if __name__ == "__main__":
    unittest.main()
