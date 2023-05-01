#! /usr/bin/env python


from random import Random
from typing import Literal
from uuid import UUID

import numpy as np

from gnn_devtools.utils import _diag_indices_3D, _triu_indices_3D


def mkAdjacencyMatrix(num_nodes:int, num_relations:int, undirected:bool,
                      reflexive:bool, p_relation:list|np.ndarray,
                      rng:np.random.Generator) -> np.ndarray:
    """ Generate random adjacency matrix

    :param num_nodes: total number of nodes in the graph
    :param num_relations: total number of unique relations in the graph
    :param undirected: whether the edges are directed or not
    :param reflexive: whether nodes can refer to themselves
    :param p_relation: the probability for each node of being connected via
        a relation. If num_relations > 1 then this is expected to be a list
        with as many values as there are relations. 
    :param rng: Numpy random number generator
    :returns: Numpy array of size RxNxN or NxN
    """
    if num_relations == 1:
        assert len(p_relation) >= 1, ""
        A = rng.binomial(n = 1, p = p_relation[0],
                         size = (num_nodes, num_nodes)).astype(np.int8)
    elif num_relations >= 2:
        assert num_relations < 256, "Expects num_relations < 256"
        A = np.zeros((num_relations, num_nodes, num_nodes)).astype(np.int8)

        assert len(p_relation) == num_relations
        for i in range(A.shape[0]):
            A[i] = rng.binomial(n = 1, p = p_relation[i],
                     size = (num_nodes, num_nodes)).astype(np.int8)
    else:
        raise Exception("Expects 0 < num_relations < 256")

    # ensure no disconnected nodes
    A = _mkConnected(A, undirected = undirected, rng = rng)

    if undirected:
        # ensure that A[i, j] == A[j, i]
        if A.ndim == 3:
            A[_triu_indices_3D(num_nodes, num_relations, k = 1)] =\
            A.transpose(0, 2, 1)[_triu_indices_3D(num_nodes, num_relations,
                                                  k = 1)]
        elif A.ndim == 2:
            A[np.triu_indices(num_nodes, k = 1)] =\
            A.T[np.triu_indices(num_nodes, k = 1)]
        else:
            raise Exception("Expects 2D or 3D array")
    
    if A.ndim == 3:
        A[_diag_indices_3D(num_nodes, num_relations)] = 1 if reflexive else 0
    elif A.ndim == 2:
        A[np.diag_indices(num_nodes)] = 1 if reflexive else 0
    else:
        raise Exception("Expects 2D or 3D array")

    return A

def _mkConnected(A:np.ndarray, undirected:bool,
                 rng:np.random.Generator) -> np.ndarray:
    """ Ensure no disconnected nodes

    :param A: an RxNxN or NxN adjacency matrix
    :param undirected: whether the edges are directed or not
    :param rng: Numpy random number generator
    :returns: Numpy array of size RxNxN or NxN
    """
    num_nodes = A.shape[-1]
    if undirected:
        r_sum = A.sum(axis = A.ndim - 1)
    else:
        r_sum = A.sum(axis = A.ndim - 1) + A.sum(axis = A.ndim - 2)

    r_idx = np.where(r_sum == 0)
    node_idx = [rng.integers(low = 0, high = num_nodes)
                for _ in range(r_idx[0].shape[0])]
    A[(*r_idx, node_idx)] = 1

    return A

def mkLabels(num_labels:int, pre_char:str, rng:Random) -> np.ndarray:
    """ Generate random labels

    :param num_labels: total number of labels to generate
    :param pre_char: prepend this character to all labels
    :param rng: Python random number generator
    :returns: Numpy array of size num_labels
    """
    pre_char = '' if pre_char is None else pre_char
    labels = np.empty(num_labels, dtype=object)
    for i in range(num_labels):
        labels[i] = pre_char + UUID(int=rng.getrandbits(128)).hex

    return labels

def mkNodeEmbeddings(num_nodes:int, mode:Literal["one-hot", "random"],
                     embedding_dim:int,
                     rng:np.random.Generator) -> np.ndarray:
    """ Generate random node embedding matrix

    :param num_nodes: total number of nodes in the graph
    :param mode: use one-hot encoded or random embeddings
    :param embedding_dim: size (width) of embeddings if mode is random
    :param rng: NumPy random number generator
    :returns: NxEmb Numpy array
    """
    assert num_nodes > 0
    if mode == "one-hot":
        E = np.eye(num_nodes, dtype=np.int8)
    elif mode == "random":
        assert embedding_dim > 0
        E = rng.random((num_nodes, embedding_dim))
    else:
        raise Exception("Expects 'one-hot' or 'random'")

    return E

def mkFeatureEmbeddings(num_nodes:int, features:list[dict],
                        rng:np.random.Generator) -> tuple[np.ndarray,
                                                          np.ndarray]:
    """ Generate random feature embedding matrix with one or more features

    :param num_nodes: total number of nodes in the graph
    :param features: list of dictionaries, one per feature,
        with parameters specifying how to generate the values.
        Necessary keys are 'mode', 'embedding_dim', 'distribution',
        and 'p'.
    :param rng: NumPy random number generator
    :returns: tuple with Numpy array of size Nx(Emb+F) containing features
        and a Numpy array of size F with index pointers to the start of
        each block.
    """
    num_features = len(features)
    assert num_features > 0

    embedding_dim = sum([f['embedding_dim'] for f in features])
    F = np.zeros((num_nodes, embedding_dim))
    F_idxptr = np.zeros(num_features)

    i, j = 0, 0
    for f_idx in range(num_features):
        F_idxptr[f_idx] = i

        embedding_dim = features[f_idx]['embedding_dim']
        j += embedding_dim

        F[:, i:j] = mkFeatureEmbedding(num_nodes=num_nodes,
                                       rng=rng, **features[f_idx])

        i = j

    return (F, F_idxptr)

def mkFeatureEmbedding(num_nodes:int, mode:Literal["natural", "real"],
                       embedding_dim:int, distribution:Literal["random",
                       "normal", "exponential", "poisson", "uniform"],
                       p:float, rng:np.random.Generator,
                       **kwargs:dict|None) -> np.ndarray:
    """ Generate random feature embedding matrix with one feature

    :param num_nodes: total number of nodes in the graph
    :param mode: generate natural numbers or real numbers
    :param embedding_dim: size (width) of embeddings
    :param distribution: distribution to sample from
    :param p: probability of a node having this feature
    :param rng: NumPy random number generator
    :returns: NxEmb Numpy array
    """
    assert num_nodes > 0
    assert embedding_dim > 0

    dist = getattr(rng, distribution)
    F = dist(size = (num_nodes, embedding_dim), **kwargs)
    if mode == "natural":
        F = F.astype(int)
    elif mode == "real":
        F = F.astype(float)
    else:
        raise Exception("Expects 'natural' or 'real'")

    if p > 0.:
        mask = rng.binomial(n = 1, p = p,
                            size = num_nodes).astype(bool)
        F[~mask, :] = 0.

    return F

def mkDegreeMatrix(A:np.ndarray, undirected:bool, 
                   collapsed:bool) -> np.ndarray:
    """ Generate degree matrix

    :param A: a 2D or 3D adjacency matrix
    :param undirected: whether the graph is directed or not
    :param collapsed: collapse the degree matrices from different
        relations
    :returns: RxNxN or NxN Numpy array
    """
    
    D = _mkInOutDegreeMatrix(A, mode = "out", collapsed = collapsed)
    if not undirected:
        D += _mkInOutDegreeMatrix(A, mode = "in", collapsed = collapsed)
    else:
        # count reflexive edges twice
        num_nodes = A.shape[-1]
        num_relations = 1 if A.ndim == 2 else A.shape[0]
        if A.ndim == 2:
            diag_idx = np.diag_indices(num_nodes)
            D[diag_idx] += A[diag_idx]
        elif A.ndim == 3:
            if collapsed:
                diag_idx = np.diag_indices(num_nodes)
                D[diag_idx] += A.sum(axis = 0)[diag_idx]
            else:
                diag_idx = _diag_indices_3D(num_nodes, num_relations)
                D[diag_idx] += A[diag_idx].flatten()
        else:
            raise Exception()

    return D

def _mkInOutDegreeMatrix(A:np.ndarray, mode:Literal["in", "out"],
                         collapsed:bool) -> np.ndarray:
    """ Generate in- or out- degree matrix

    :param A: a 2D or 3D adjacency matrix
    :param mode: compute indegree or outdegree
    :param collapsed: collapse the degree matrices from different
        relations if |R| > 1
    :returns: RxNxN or NxN Numpy array
    """
    assert 2 <= A.ndim <= 3

    num_nodes = A.shape[-1]
    num_relations = 1 if A.ndim == 2 else A.shape[0]
    if mode == "out":
        dim = A.ndim - 1
    elif mode == "in":
        dim = A.ndim - 2
    else:
        raise Exception()

    degrees = A.sum(axis = dim)
    D = np.zeros(A.shape, dtype = int)
    if A.ndim == 2:
        diag_idx = np.diag_indices(num_nodes)
        D[diag_idx] = degrees
    elif A.ndim == 3:
        diag_idx = _diag_indices_3D(num_nodes, num_relations)
        D[diag_idx] = degrees.flatten()

        if collapsed:
            D = D.sum(axis = 0)
    else:
        raise Exception()

    return D

def mkIncidenceMatrix(A:np.ndarray, undirected:bool,
                      relation:int|None) -> np.ndarray:
    """ Generate incidence matrix

    :param A: a 2D or 3D adjacency matrix
    :param undirected: whether the graph is directed or not
    :param relation: only consider the slice belonging to this
        relation index if A is 3D
    :returns: NxE Numpy array
    """
    num_nodes = A.shape[-1]
    if relation is not None:
        assert A.ndim == 3 and 0 <= relation < A.shape[0]
        # look only at a specific relation
        A = A[relation]

    if A.ndim == 2: 
        B = _mkIncidenceMatrix2D(A, undirected)
    elif A.ndim == 3:
        num_relations = A.shape[0]
        num_edges = A.sum() if not undirected else\
            (A.sum() + A[_diag_indices_3D(num_nodes, num_relations)].sum())//2

        B = np.zeros((num_nodes, num_edges), dtype = np.int8)
        i, j = 0, 0
        for r in range(num_relations):
            n = A[r].sum() if not undirected else\
                (A[r].sum() + A[r][np.diag_indices(num_nodes)].sum())//2
            j += n

            B[:, i:j] = _mkIncidenceMatrix2D(A[r], undirected = undirected)

            i = j
    else:
        raise Exception()

    return B

def _mkIncidenceMatrix2D(A:np.ndarray, undirected:bool) -> np.ndarray:
    """ Generate incidence matrix for 2D adjacency matrix

    :param A: a 2D adjacency matrix
    :param undirected: whether the graph is directed or not
    :returns: NxE Numpy array
    """
    assert A.ndim == 2
    if undirected:
        return _mkIncidenceMatrixUndirected2D(A)
    else:
        return _mkIncidenceMatrixDirected2D(A)

def _mkIncidenceMatrixDirected2D(A:np.ndarray) -> np.ndarray:
    """ Generate incidence matrix for directed 2D adjacency matrix

    :param A: a 2D adjacency matrix
    :returns: NxE Numpy array
    """
    num_nodes = A.shape[-1]
    num_edges = A.sum()
    edge_idx = np.arange(num_edges)

    B = np.zeros((num_nodes, num_edges), dtype = np.int8)

    node_out_idx, node_in_idx = np.where(A == 1)
    B[node_out_idx, edge_idx] = -1
    B[node_in_idx, edge_idx] = 1
    
    # ensure 2 on loop
    loop_idx = np.where(node_out_idx-node_in_idx == 0)[0]
    B[node_out_idx[loop_idx], edge_idx[loop_idx]] = 2

    return B

def _mkIncidenceMatrixUndirected2D(A:np.ndarray) -> np.ndarray:
    """ Generate incidence matrix for indirected 2D adjacency matrix

    :param A: a 2D adjacency matrix
    :returns: NxE Numpy array
    """
    num_nodes = A.shape[-1]
    num_edges = (A.sum() + A[np.diag_indices(num_nodes)].sum()) // 2
    edge_idx = np.arange(num_edges)
    triu_idx = np.triu_indices(num_nodes)
    
    B = np.zeros((num_nodes, num_edges), dtype = np.int8)

    idx = np.where(A[triu_idx] == 1)
    B[triu_idx[0][idx], edge_idx] = 1
    B[triu_idx[1][idx], edge_idx] += 1  # ensure 2 on loop

    return B

def mkSymmetric(A:np.ndarray) -> np.ndarray:
    """ Transform a directed adjacency matrix to an undirected one

    :param A: a 2D or 3D adjacency matrix
    :returns: RxNxN or NxN Numpy array
    """
    A = A.astype(bool)
    
    if A.ndim == 2:
        return A + A.T
    else:
        assert A.ndim == 3
        return A + A.transpose(0, 2, 1)
