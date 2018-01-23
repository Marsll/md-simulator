from ..cell_order import create_cell_order_3d, create_cell_order_2d
import numpy as np
import numpy.testing as npt


"""Tests"""


def test_cell_order_2d():
    """Test for 6 cells in 2d"""
    cell_order_reference = [[1, 3, 4], [2, 3, 4, 5], [4, 5], [4], [5], []]
    cell_order = create_cell_order_2d(1, [3, 2])
    npt.assert_array_equal(cell_order_reference, cell_order)


def test_cell_order_2d_one_cell():
    """Test for 6 cells in 2d"""
    cell_order_reference = [[]]
    cell_order = create_cell_order_2d(1, [1, 1])
    npt.assert_array_equal(cell_order_reference, cell_order)


test_cell_order_2d()
test_cell_order_2d_one_cell()


def test_cell_order_3d():
    """Test for 8 cells in 2d"""
    cell_order_reference = [[1, 2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7], [
        3, 4, 5, 6, 7], [4, 5, 6, 7], [5, 6, 7], [6, 7], [7], []]
    cell_order = create_cell_order_3d(1, [2, 2, 2])
    npt.assert_array_equal(cell_order_reference, cell_order)


def test_cell_order_3d_one_cell():
    """Test for 6 cells in 2d"""
    cell_order_reference = [[]]
    cell_order = create_cell_order_3d(1, [1, 1, 1])
    npt.assert_array_equal(cell_order_reference, cell_order)


test_cell_order_3d()
test_cell_order_3d_one_cell()

print("tests done")


