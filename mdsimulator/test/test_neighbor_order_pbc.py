"""Tests for neighbor_order_pbc."""

import numpy as np
import numpy.testing as npt
from ..neighbor_order_pbc import get_neighbors


def test_nb_order_pbc_2x2():
    n_cells = np.array([2, 2])
    total_n_cells = np.prod(n_cells)
    box = np.arange(total_n_cells)
    box = box.reshape(n_cells)

    ref_nbs = [np.array([1, 2, 3]), np.array([2, 3]), np.array([3])]
    nbs = get_neighbors(box)
    for arr_ref, arr in zip(ref_nbs, nbs):
        npt.assert_equal(arr_ref, arr)


def test_nb_order_pbc_3x3():
    n_cells = np.array([3, 3])
    total_n_cells = np.prod(n_cells)
    box = np.arange(total_n_cells)
    box = box.reshape(n_cells)

    ref_nbs = [np.array([1, 2, 3, 4, 5, 6, 7, 8]),
               np.array([2, 3, 4, 5, 6, 7, 8]),
               np.array([3, 4, 5, 6, 7, 8]),
               np.array([4, 5, 6, 7, 8]),
               np.array([5, 6, 7, 8]),
               np.array([6, 7, 8]),
               np.array([7, 8]),
               np.array([8]), ]
    nbs = get_neighbors(box)
    for arr_ref, arr in zip(ref_nbs, nbs):
        npt.assert_equal(arr_ref, np.sort(arr))


def test_nb_order_pbc_4x4():
    n_cells = np.array([4, 4])
    total_n_cells = np.prod(n_cells)
    box = np.arange(total_n_cells)
    box = box.reshape(n_cells)

    ref_nbs = [np.array([1, 3, 4, 5, 7, 12, 13, 15]),
               np.array([2, 4, 5, 6, 12, 13, 14]),
               np.array([3, 5, 6, 7, 13, 14, 15]),
               np.array([4, 6, 7, 12, 14, 15]),
               np.array([5, 7, 8, 9, 11]),
               np.array([6, 8, 9, 10]),
               np.array([7, 9, 10, 11]),
               np.array([8, 10, 11]),
               np.array([9, 11, 12, 13, 15]),
               np.array([10, 12, 13, 14]),
               np.array([11, 13, 14, 15]),
               np.array([12, 14, 15]),
               np.array([13, 15]),
               np.array([14]),
               np.array([15])]
    nbs = get_neighbors(box)
    for arr_ref, arr in zip(ref_nbs, nbs):
        npt.assert_equal(arr_ref, np.sort(arr))


def test_nb_order_pbc_4x4x2():
    # First index - z, Second index - y, Third index - x
    n_cells = np.array([2, 4, 4])
    total_n_cells = np.prod(n_cells)
    box = np.arange(total_n_cells)
    box = box.reshape(n_cells)

    ref_zero = np.array([1, 3, 4, 5, 7, 12, 13, 15, 16,
                         17, 19, 20, 21, 23, 28, 29, 31])
    ref_12 = np.array([13, 15, 16, 17, 19, 24, 25, 27, 28, 29, 31])
    ref_26 = np.array([27, 29, 30, 31])
    ref_30 = np.array([31])

    nbs = get_neighbors(box)
    npt.assert_equal(ref_zero, nbs[0])
    npt.assert_equal(ref_12, nbs[12])
    npt.assert_equal(ref_26, nbs[26])
    npt.assert_equal(ref_30, nbs[-1])


test_nb_order_pbc_2x2()
test_nb_order_pbc_3x3()
test_nb_order_pbc_4x4()
test_nb_order_pbc_4x4x2()
