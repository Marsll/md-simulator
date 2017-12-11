import numpy as np
import numpy.testing as npt
from ..neighbor_list import NeighborList


def test1_cell_index2D():
    dim_box = np.array([3, 3])
    pos = np.array([[0, 0], [2, 2], [1.3, 2.6]])
    cell_width = 1

    nl = NeighborList(dim_box, pos, cell_width)
    idx1 = nl.get_cell_index(0)
    idx2 = nl.get_cell_index(1)
    idx3 = nl.get_cell_index(2)

    reference_idx1 = 0
    reference_idx2 = 8
    reference_idx3 = 7

    npt.assert_equal(idx1, reference_idx1)
    npt.assert_equal(idx2, reference_idx2)
    npt.assert_equal(idx3, reference_idx3)

def test2_cell_index2D():
    dim_box = np.array([5.2, 3.2])
    pos = np.array([[1.1, 1.1], [4.4, 2.0], [5.0, 2.6]])
    cell_width = 1.2

    nl = NeighborList(dim_box, pos, cell_width)
    idx1 = nl.get_cell_index(0)
    idx2 = nl.get_cell_index(1)
    idx3 = nl.get_cell_index(2)

    reference_idx1 = 0
    reference_idx2 = 7
    reference_idx3 = 7

    npt.assert_equal(idx1, reference_idx1)
    npt.assert_equal(idx2, reference_idx2)
    npt.assert_equal(idx3, reference_idx3)

def test_cell_index3D():
    dim_box = np.array([100, 100, 100])
    pos = np.array([[10, 40, 80], [5, 0, 12], [31.3, 50.6, 70.4]])
    cell_width = 10

    nl = NeighborList(dim_box, pos, cell_width)
    idx1 = nl.get_cell_index(0)
    idx2 = nl.get_cell_index(1)
    idx3 = nl.get_cell_index(2)

    reference_idx1 = 841
    reference_idx2 = 100
    reference_idx3 = 753

    npt.assert_equal(idx1, reference_idx1)
    npt.assert_equal(idx2, reference_idx2)
    npt.assert_equal(idx3, reference_idx3)



def test_neighbor_list1D():
    dim_box = 2
    pos = np.array([[0.2], [0.3], [1.0], [1.8]])
    cell_width = 1

    nl = NeighborList(dim_box, pos, cell_width)

    reference_nl_head = np.array([1, 3])
    reference_nl_list = np.array([-1, 0, -1, 2])

    npt.assert_equal(nl.head, reference_nl_head)
    npt.assert_equal(nl.list, reference_nl_list)


def test_neighbor_list2D():
    dim_box = (4, 4)
    pos = np.array([[3.6, 1.6], [3.5, 1.9], [0.2, 0.8], [1.4, 1.7], [1, 2]])
    cell_width = 1.5

    nl = NeighborList(dim_box, pos, cell_width)

    reference_nl_head = np.array([2, -1, 4, 1])
    reference_nl_list = np.array([-1, 0, -1, -1, 3])

    npt.assert_equal(nl.head, reference_nl_head)
    npt.assert_equal(nl.list, reference_nl_list)
