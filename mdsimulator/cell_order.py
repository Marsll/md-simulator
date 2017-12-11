import numpy as np
import numpy.testing as npt

def create_cell_order_2d(r_cut, dims):
    """Returns array with all neighbors of each cell"""
    size = np.empty(len(dims))
    for i, dim in enumerate(dims):
        size[i] = np.int(dim / r_cut)
    order = []
    for j in np.arange(0, size[1]):
        for i in np.arange(0, size[0]):    
            index = i + j * size[0]
            nb = []
            if i + 1 < size[0]:
                nb += [index + 1]
            if j + 1 < size[1]:
                if i != 0:
                    nb += [index + size[0] - 1]
                nb += [index + size[0]]
                if i + 1 < size[0]:
                    nb += [index + size[0] + 1]
            order += [nb]
    return order
            
    
def create_cell_order_3d(r_cut, dims):
    """Returns array with all neighbors of each cell"""
    size = np.empty(len(dims))
    for i, dim in enumerate(dims):
        size[i] = np.int(dim / r_cut)
    order = []
    for k in np.arange(0, size[2]):
        for j in np.arange(0, size[1]):
            for i in np.arange(0, size[0]):    
                index = i + j * size[0] + k * size[0] * size[1]
                nb = []
                
                if i + 1 < size[0]:
                    nb += [index + 1]
                    
                if j + 1 < size[1]:
                    if i != 0:
                        nb += [index + size[0] - 1]
                    nb += [index + size[0]]
                    if i + 1 < size[0]:
                        nb += [index + size[0] + 1]
                        
                if k + 1 < size[2]:
                    if j != 0:
                        if i != 0:
                            nb += [index - size[0] - 1 + size[0] * size[1]]
                        nb += [index - size[0] + size[0] * size[1]]
                        if i + 1 < size[0]:
                            nb += [index - size[0] + 1 + size[0] * size[1]]
                    if i != 0:
                        nb += [index - 1 + size[0] * size[1]]
                    nb += [index + size[0] * size [1]]                                    
                    if i + 1 < size[0]:
                        nb += [index + 1 + size[0] * size[1]]

                            
                    if j + 1 < size[1]:
                        if i != 0:
                            nb += [index + size[0] - 1 + size[0] * size[1]]
                        nb += [index + size[0] + size[0] * size[1]]
                        if i + 1 < size[0]:
                            nb += [index + size[0] + 1 + size[0] * size[1]]    
                    
                    
                    
                order += [nb]
    return order       
    
    
"""Tests"""

def test_cell_order_2d():
    """Test for 6 cells in 2d"""
    cell_order_reference = [[1,3,4], [2,3,4,5], [4,5], [4], [5], []]
    cell_order = create_cell_order_2d(1, [3,2])
    npt.assert_array_equal(cell_order_reference, cell_order )
    
def test_cell_order_2d_one_cell():
    """Test for 6 cells in 2d"""
    cell_order_reference = [ []]
    cell_order = create_cell_order_2d(1, [1,1])
    npt.assert_array_equal(cell_order_reference, cell_order )
    
test_cell_order_2d()
test_cell_order_2d_one_cell()

def test_cell_order_3d():
    """Test for 8 cells in 2d"""
    cell_order_reference = [[1,2,3,4,5,6,7], [2,3,4,5,6,7], [3,4,5,6,7], [4,5,6,7], [5,6,7], [6,7], [7], []]
    cell_order = create_cell_order_3d(1, [2, 2, 2])
    npt.assert_array_equal(cell_order_reference, cell_order )
    
    
def test_cell_order_3d_one_cell():
    """Test for 6 cells in 2d"""
    cell_order_reference = [ []]
    cell_order = create_cell_order_3d(1, [1, 1, 1])
    npt.assert_array_equal(cell_order_reference, cell_order )

test_cell_order_3d()    
test_cell_order_3d_one_cell()

print("tests done")