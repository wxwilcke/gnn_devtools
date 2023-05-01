#! /usr/bin/env python

import numpy as np


def _triu_indices_3D(num_nodes:int, num_relations:int,
                     k:int) -> tuple[np.ndarray,
                                     np.ndarray,
                                     np.ndarray]:
    """ Upper triangle indices of 3D array
    
    :param num_nodes: total number of nodes in the graph
    :param num_relations: total number of unique relations in the graph
    :param k: steps from diagonal
    :returns: tuple with z, x, y coordinates as Numpy arrays
    """
    assert num_nodes > 0
    assert num_relations > 0

    x, y = np.triu_indices(num_nodes, k = k)
    l = x.shape[0]

    x = np.tile(x, num_relations)
    y = np.tile(y, num_relations)
    z = np.repeat(np.arange(num_relations), l)

    return (z, x, y)

def _tril_indices_3D(num_nodes:int, num_relations:int,
                     k:int) -> tuple[np.ndarray,
                                     np.ndarray,
                                     np.ndarray]:
    """ Lower triangle indices of 3D array
    
    :param num_nodes: total number of nodes in the graph
    :param num_relations: total number of unique relations in the graph
    :param k: steps from diagonal
    :returns: tuple with z, x, y coordinates as Numpy arrays
    """
    assert num_nodes > 0
    assert num_relations > 0

    x, y = np.triu_indices(num_nodes, k = k)
    l = x.shape[0]

    x = np.tile(x, num_relations)
    y = np.tile(y, num_relations)
    z = np.repeat(np.arange(num_relations), l)

    return (z, x, y)

def _diag_indices_3D(num_nodes:int, num_relations:int) -> tuple[np.ndarray,
                                                                np.ndarray,
                                                                np.ndarray]:
    """ Diagonal indices of 3D array
    
    :param num_nodes: total number of nodes in the graph
    :param num_relations: total number of unique relations in the graph
    :returns: tuple with z, x, y coordinates as Numpy arrays
    """
    assert num_nodes > 0
    assert num_relations > 0

    xy = np.tile(np.arange(num_nodes), num_relations)
    z = np.repeat(np.arange(num_relations), num_nodes)

    return (z, xy, xy)

def _eye_3D(num_nodes:int, num_relations:int) -> np.ndarray:
    """ Create a 3D array with ones on the diagonal
    
    :param num_nodes: total number of nodes in the graph
    :param num_relations: total number of unique relations in the graph
    :returns: R x N x N Numpy array
    """
    shape = (num_relations, num_nodes, num_nodes)

    eye = np.zeros(shape, dtype = np.int8)
    eye[_diag_indices_3D(num_nodes, num_relations)] = 1

    return eye
