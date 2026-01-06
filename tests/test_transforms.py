"""unit tests for transforms"""

import unittest

import torch

from radiomana.transforms import LogNoise, RandomTimeCrop


class TestCustomTransforms(unittest.TestCase):
    def test_log_noise(self):
        """is LogNoise custom transform working as expected?"""
        transform = LogNoise(noise_power_db=-90, p=0.5)
        # test with (tensor, target) tuple
        batch = (torch.randn(16, 512, 243), torch.randint(0, 9, (16,)))
        output = transform(batch)

        self.assertEqual(batch[0].shape, output[0].shape)
        # ensure labels are unchanged
        self.assertTrue(torch.equal(batch[1], output[1]))
        # ensure no nan values in output
        self.assertFalse(torch.isnan(output[0]).any())

    def test_random_time_crop(self):
        """is RandomTimeCrop custom transform working as expected?"""
        crop_width = 211
        transform = RandomTimeCrop(crop_width=crop_width)
        # test with (tensor, target) tuple
        batch = (torch.randn(16, 512, 243), torch.randint(0, 9, (16,)))
        output = transform(batch)
        # ensure non-time dimensions are unchanged
        self.assertEqual(batch[0].shape[0], output[0].shape[0])
        self.assertEqual(batch[0].shape[1], output[0].shape[1])
        # ensure time dimension is cropped to fixed width
        self.assertEqual(output[0].shape[2], crop_width)
        # ensure labels are unchanged
        self.assertTrue(torch.equal(batch[1], output[1]))

        # test with smaller input (should return unchanged)
        small_batch = (torch.randn(16, 512, 100), torch.randint(0, 9, (16,)))
        small_output = transform(small_batch)
        self.assertEqual(small_output[0].shape[2], 100)  # unchanged
