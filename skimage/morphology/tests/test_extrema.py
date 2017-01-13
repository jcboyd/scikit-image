import math
import unittest

import numpy as np

from skimage.morphology import extrema

eps = 1e-12


def diff(a, b):
    a = np.asarray(a)
    a = a.astype(np.float64)
    b = np.asarray(b)
    b = b.astype(np.float64)
    t = ((a - b)**2).sum()
    return math.sqrt(t)


class TestExtrema(unittest.TestCase):

    def test_saturated_arithmetic(self):
        "Adding/subtracting a constant and clipping"
        # Test for unsigned integer
        data = np.array([[250, 251, 5, 5],
                         [100, 200, 253, 252],
                         [4, 10, 1, 3]],
                        dtype=np.uint8)
        # adding the constant
        img_constant_added = extrema._add_constant_clip(data, 4)
        expected = np.array([[254, 255, 9, 9],
                             [104, 204, 255, 255],
                             [8, 14, 5, 7]],
                            dtype=np.uint8)
        error = diff(img_constant_added, expected)
        assert error < eps
        img_constant_subtracted = extrema._subtract_constant_clip(data, 4)
        expected = np.array([[246, 247, 1, 1],
                             [96, 196, 249, 248],
                             [0, 6, 0, 0]],
                            dtype=np.uint8)
        error = diff(img_constant_subtracted, expected)
        assert error < eps

        # Test for signed integer
        data = np.array([[32767, 32766],
                         [-32768, -32767]],
                        dtype=np.int16)
        img_constant_added = extrema._add_constant_clip(data, 1)
        expected = np.array([[32767, 32767],
                             [-32767, -32766]],
                            dtype=np.int16)
        error = diff(img_constant_added, expected)
        assert error < eps
        img_constant_subtracted = extrema._subtract_constant_clip(data, 1)
        expected = np.array([[32766, 32765],
                             [-32768, -32768]],
                            dtype=np.int16)
        error = diff(img_constant_subtracted, expected)
        assert error < eps

    def test_local_maxima(self):
        "local maxima for various data types"
        data = np.array([[10,  11,  13,  14,  14,  15,  14,  14,  13,  11],
                         [11,  13,  15,  16,  16,  16,  16,  16,  15,  13],
                         [13,  15,  40,  40,  18,  18,  18,  60,  60,  15],
                         [14,  16,  40,  40,  19,  19,  19,  60,  60,  16],
                         [14,  16,  18,  19,  19,  19,  19,  19,  18,  16],
                         [15,  16,  18,  19,  19,  20,  19,  19,  18,  16],
                         [14,  16,  18,  19,  19,  19,  19,  19,  18,  16],
                         [14,  16,  80,  80,  19,  19,  19, 100, 100,  16],
                         [13,  15,  80,  80,  18,  18,  18, 100, 100,  15],
                         [11,  13,  15,  16,  16,  16,  16,  16,  15,  13]],
                        dtype=np.uint8)
        expected_result = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                                   dtype=np.uint8)
        for dtype in [np.uint8, np.uint64, np.int8, np.int64,
                      np.float, np.double]:

            test_data = data.astype(dtype)
            out = extrema.local_maxima(test_data)

            error = diff(expected_result, out)
            assert error < eps
            assert out.dtype == expected_result.dtype

    def test_local_minima(self):
        "local minima for various data types"

        data = np.array([[10,  11,  13,  14,  14,  15,  14,  14,  13,  11],
                         [11,  13,  15,  16,  16,  16,  16,  16,  15,  13],
                         [13,  15,  40,  40,  18,  18,  18,  60,  60,  15],
                         [14,  16,  40,  40,  19,  19,  19,  60,  60,  16],
                         [14,  16,  18,  19,  19,  19,  19,  19,  18,  16],
                         [15,  16,  18,  19,  19,  20,  19,  19,  18,  16],
                         [14,  16,  18,  19,  19,  19,  19,  19,  18,  16],
                         [14,  16,  80,  80,  19,  19,  19, 100, 100,  16],
                         [13,  15,  80,  80,  18,  18,  18, 100, 100,  15],
                         [11,  13,  15,  16,  16,  16,  16,  16,  15,  13]],
                        dtype=np.uint8)
        data = 100 - data
        expected_result = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                                   dtype=np.uint8)
        for dtype in [np.uint8, np.uint64, np.int8, np.int64,
                      np.float, np.double]:
            data = data.astype(dtype)
            out = extrema.local_minima(data)

            error = diff(expected_result, out)
            assert error < eps
            assert out.dtype == expected_result.dtype

    def test_h_maxima(self):
        "h-maxima for various data types"

        data = np.array([[10,  11,  13,  14,  14,  15,  14,  14,  13,  11],
                         [11,  13,  15,  16,  16,  16,  16,  16,  15,  13],
                         [13,  15,  40,  40,  18,  18,  18,  60,  60,  15],
                         [14,  16,  40,  40,  19,  19,  19,  60,  60,  16],
                         [14,  16,  18,  19,  19,  19,  19,  19,  18,  16],
                         [15,  16,  18,  19,  19,  20,  19,  19,  18,  16],
                         [14,  16,  18,  19,  19,  19,  19,  19,  18,  16],
                         [14,  16,  80,  80,  19,  19,  19, 100, 100,  16],
                         [13,  15,  80,  80,  18,  18,  18, 100, 100,  15],
                         [11,  13,  15,  16,  16,  16,  16,  16,  15,  13]],
                        dtype=np.uint8)

        expected_result = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                                   dtype=np.uint8)
        for dtype in [np.uint8, np.uint64, np.int8, np.int64,
                      np.float, np.double]:
            data = data.astype(dtype)
            out = extrema.h_maxima(data, 40)

            error = diff(expected_result, out)
            assert error < eps

    def test_h_minima(self):
        "h-minima for various data types"

        data = np.array([[10,  11,  13,  14,  14,  15,  14,  14,  13,  11],
                         [11,  13,  15,  16,  16,  16,  16,  16,  15,  13],
                         [13,  15,  40,  40,  18,  18,  18,  60,  60,  15],
                         [14,  16,  40,  40,  19,  19,  19,  60,  60,  16],
                         [14,  16,  18,  19,  19,  19,  19,  19,  18,  16],
                         [15,  16,  18,  19,  19,  20,  19,  19,  18,  16],
                         [14,  16,  18,  19,  19,  19,  19,  19,  18,  16],
                         [14,  16,  80,  80,  19,  19,  19, 100, 100,  16],
                         [13,  15,  80,  80,  18,  18,  18, 100, 100,  15],
                         [11,  13,  15,  16,  16,  16,  16,  16,  15,  13]],
                        dtype=np.uint8)
        data = 100 - data
        expected_result = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                                   dtype=np.uint8)
        for dtype in [np.uint8, np.uint64, np.int8, np.int64,
                      np.float, np.double]:
            data = data.astype(dtype)
            out = extrema.h_minima(data, 40)

            error = diff(expected_result, out)
            assert error < eps
            assert out.dtype == expected_result.dtype

    def test_extrema_float(self):
        "specific tests for float type"
        print 'test_extrema_float'
        data = np.array([[10,  11,  13,  14,  14,  15,  14,  14,  13,  11],
                         [11,  13,  15,  16,  16,  16,  16,  16,  15,  13],
                         [13,  15,  40,  40,  18,  18,  18,  60,  60,  15],
                         [14,  16,  40,  40,  19,  19,  19,  60,  60,  16],
                         [14,  16,  18,  19,  19,  19,  19,  19,  18,  16],
                         [15,  18.2,  18,  19,  20.4,  20,  19,  19,  18,  16],
                         [14,  16,  18,  19,  19,  19,  19,  19,  18,  16],
                         [14,  16,  80,  80,  19,  19,  19, 100, 100,  16],
                         [13,  15,  80,  80,  18,  18,  18, 100, 100,  15],
                         [11,  13,  15,  16,  16,  16,  16,  16,  15,  13]],
                        dtype=np.float32)
        inverted_data = 200 - data

        expected_result = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                                   dtype=np.uint8)

        # test for local maxima with automatic step calculation
        out = extrema.local_maxima(data)
        error = diff(expected_result, out)
        assert error < eps

        # test for local minima with automatic step calculation
        out = extrema.local_minima(inverted_data)
        error = diff(expected_result, out)
        assert error < eps

        out = extrema.h_maxima(data, 0.3)
        expected_result = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                                   dtype=np.uint8)

        error = diff(expected_result, out)
        assert error < eps

        out = extrema.h_minima(inverted_data, 0.3)
        error = diff(expected_result, out)
        assert error < eps

if __name__ == "__main__":
    np.testing.run_module_suite()
