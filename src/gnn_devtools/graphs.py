#! /usr/bin/env python


import random
from typing import Literal

import numpy as np

from gnn_devtools.matrices import (mkAdjacencyMatrix, mkDegreeMatrix,
                                   mkIncidenceMatrix, _mkInOutDegreeMatrix,
                                   mkLabels, mkNodeEmbeddings, mkSymmetric,
                                   mkFeatureEmbeddings)
from gnn_devtools.utils import _triu_indices_3D, _diag_indices_3D


class Graph():
    """ Base graph class
    """

    def __init__(self, num_nodes:int, num_relations:int,
                 p_relation:float|list|np.ndarray, undirected:bool,
                 reflexive:bool, seed:int|None = None) -> None:
        """
        :param num_nodes: total number of nodes in the graph
        :param num_relations: total number of unique relations in the graph
        :param undirected: whether the edges are directed or not
        :param reflexive: whether nodes can refer to themselves
        :param p_relation: the probability for each node of being connected via
            a relation. If num_relations > 1 then this is expected to be a list
            with as many values as there are relations. 
        :param seed: seed for random number generator
        :returns: None
        """
        assert num_nodes >= 2
        assert num_relations >= 1

        if type(p_relation) is float:
            p_relation: list[float] = [p_relation] * num_relations

        self._rd = random.Random()
        self._rd.seed(seed)
        self._rng = np.random.default_rng(seed)

        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.p_relation = p_relation
        self.undirected = undirected
        self.reflexive = reflexive

        self.A = mkAdjacencyMatrix(num_nodes = self.num_nodes,
                                   num_relations = self.num_relations,
                                   undirected = self.undirected,
                                   reflexive = self.reflexive,
                                   p_relation = self.p_relation,
                                   rng = self._rng)

    def adjacency(self) -> np.ndarray:
        """ Return the adjacency matrix
        
        :returns: a Numpy array
        """
        return self.A

    def degree(self, mode:Literal["in", "out"]|None = None,
               collapsed:bool = True) -> np.ndarray:
        """ Return the degree matrix

        :param mode: return the in-degree, out-degree, or the merged 
            in- and outdegree matrix
        :param collapsed: collapse the degree matrices from different
            relations
        :returns: a Numpy array
        """
        if mode is None:
            return mkDegreeMatrix(self.A, undirected = self.undirected,
                                  collapsed = collapsed)
        elif mode in ["in", "out"]:
            assert not self.undirected
            return _mkInOutDegreeMatrix(self.A, mode = mode,
                                        collapsed = collapsed)
        else:
            raise Exception("Expects mode to be 'in', 'out', or 'None'")

    def incidence(self, relation:int|None = None) -> np.ndarray:
        """ Return the incidence matrix
        
        :param relation: index of relation slice in 3D matrix or None for all
        :returns: a Numpy array
        """
        return mkIncidenceMatrix(self.A, undirected = self.undirected,
                                 relation = relation)

    def laplacian(self, mode:Literal["in", "out"]|None = None,
                  symmetric:bool = False, collapsed:bool = True,
                  normalized:bool = False) -> np.ndarray:
        """ Return the Laplacian matrix
        
        :param mode: return the in-degree, out-degree, or the merged 
            in- and outdegree Laplacian matrix
        :param symmetric: force a symmetric Laplacian if the graph is
            directed
        :param collapsed: collapse the matrices from different
            relations
        :param normalized: return the normalized Laplacian
        :returns: a Numpy array
        """
        if symmetric and not self.undirected:
            A = mkSymmetric(self.A)
        else:
            A = self.A

        D = self.degree(mode = mode, collapsed = collapsed)
        if A.ndim == 3 and collapsed:
            L = D - A.sum(axis = 0)
        else:
            L = D - A

        if normalized:
            D = np.linalg.inv(D ** .5)
            L = np.eye(self.num_nodes) - np.matmul(D, np.matmul(A, D))
        
        return L

    def __len__(self):
        """ Return graph length measured in number of edges
        """
        if not self.undirected:
            return self.A.sum()
        else:
            l = 0
            if self.A.ndim == 3:
                l += self.A[_triu_indices_3D(self.num_nodes,
                                             self.num_relations, k=1)].sum()
                l += self.A[_diag_indices_3D(self.num_nodes,
                                             self.num_relations)].sum()
            else:
                l += self.A[np.triu_indices(self.num_nodes, k=1)].sum()
                l += self.A[np.diag_indices(self.num_nodes)].sum()

            return l

class UndirectedGraph(Graph):
    """ An undirected graph
    """

    def __init__(self, num_nodes:int, p_relation:float = 0.2,
                 reflexive:bool = True, seed:int|None = None) -> None:
        """
        :param num_nodes: total number of nodes in the graph
        :param reflexive: whether nodes can refer to themselves
        :param p_relation: the probability for each node of being connected via
            a relation. If num_relations > 1 then this is expected to be a list
            with as many values as there are relations. 
        :param seed: seed for random number generator
        :returns: None
        """
        assert .0 < p_relation <= 1.

        super().__init__(num_nodes = num_nodes, num_relations = 1,
                         p_relation = p_relation, undirected = True,
                         reflexive = reflexive, seed = seed)

class DirectedGraph(Graph):
    """ A directed graph
    """

    def __init__(self, num_nodes:int, p_relation:float = 0.2,
                 reflexive:bool = True, seed:int|None = None) -> None:
        """
        :param num_nodes: total number of nodes in the graph
        :param reflexive: whether nodes can refer to themselves
        :param p_relation: the probability for each node of being connected via
            a relation. If num_relations > 1 then this is expected to be a list
            with as many values as there are relations. 
        :param seed: seed for random number generator
        :returns: None
        """
        assert .0 < p_relation <= 1.

        super().__init__(num_nodes = num_nodes, num_relations = 1,
                         p_relation = p_relation, undirected = False,
                         reflexive = reflexive, seed = seed)

class LabeledGraph(Graph):
    """ A labeled graph
    """

    def __init__(self, num_nodes:int, num_relations:int = 2, 
                 p_relation:float|list|np.ndarray = 0.2,
                 undirected:bool = False, reflexive:bool = True,
                 edge_labels:bool = True, node_labels:bool = True,
                 seed:int|None = None) -> None:
        """
        :param num_nodes: total number of nodes in the graph
        :param num_relations: total number of unique relations in the graph
        :param undirected: whether the edges are directed or not
        :param reflexive: whether nodes can refer to themselves
        :param p_relation: the probability for each node of being connected via
            a relation. If num_relations > 1 then this is expected to be a list
            with as many values as there are relations.
        :param node_labels: whether to generate unique node labels
        :param edge_labels: whether to generate unique edge labels
        :param seed: seed for random number generator
        :returns: None
        """
        super().__init__(num_nodes = num_nodes, num_relations = num_relations,
                         p_relation = p_relation, undirected = undirected,
                         reflexive = reflexive, seed = seed)

        if node_labels:
            nodes = mkLabels(num_labels = num_nodes,
                             pre_char = 'N',
                             rng = self._rd)
            self.i2n = nodes
        else:
            self.i2n = np.arange(num_nodes)

        if edge_labels:
            relations = mkLabels(num_labels = num_relations,
                                 pre_char = 'r',
                                 rng = self._rd)
            self.i2r = relations
        else:
            self.i2r = np.arange(num_relations)

        self.n2i = {lab: i for i, lab in enumerate(self.i2n)}
        self.r2i = {lab: i for i, lab in enumerate(self.i2r)}

class LabeledUndirectedGraph(LabeledGraph):
    """ A labeled undirected graph
    """

    def __init__(self, num_nodes:int, num_relations:int = 2, 
                 p_relation:float|list|np.ndarray = 0.2,
                 reflexive:bool = True, edge_labels:bool = True,
                 node_labels:bool = True, seed:int|None = None) -> None:
        """
        :param num_nodes: total number of nodes in the graph
        :param num_relations: total number of unique relations in the graph
        :param reflexive: whether nodes can refer to themselves
        :param p_relation: the probability for each node of being connected via
            a relation. If num_relations > 1 then this is expected to be a list
            with as many values as there are relations.
        :param node_labels: whether to generate unique node labels
        :param edge_labels: whether to generate unique edge labels
        :param seed: seed for random number generator
        :returns: None
        """

        super().__init__(num_nodes = num_nodes, num_relations = num_relations,
                         p_relation = p_relation, undirected = True,
                         reflexive = reflexive, node_labels = node_labels,
                         edge_labels = edge_labels, seed = seed)

class LabeledDirectedGraph(LabeledGraph):
    """ A labeled directed graph
    """

    def __init__(self, num_nodes:int, num_relations:int = 2, 
                 p_relation:float|list|np.ndarray = 0.2,
                 reflexive:bool = True, edge_labels:bool = True,
                 node_labels:bool = True, seed:int|None = None) -> None:
        """
        :param num_nodes: total number of nodes in the graph
        :param num_relations: total number of unique relations in the graph
        :param reflexive: whether nodes can refer to themselves
        :param p_relation: the probability for each node of being connected via
            a relation. If num_relations > 1 then this is expected to be a list
            with as many values as there are relations.
        :param node_labels: whether to generate unique node labels
        :param edge_labels: whether to generate unique edge labels
        :param seed: seed for random number generator
        :returns: None
        """

        super().__init__(num_nodes = num_nodes, num_relations = num_relations,
                         p_relation = p_relation, undirected = False,
                         reflexive = reflexive, node_labels = node_labels,
                         edge_labels = edge_labels, seed = seed)

class KnowledgeGraph(LabeledDirectedGraph):
    """ A knowledge graph
    """

    def __init__(self, num_nodes:int, num_relations:int = 2,
                 p_relation:float|list|np.ndarray = 0.2,
                 num_attributes:int = 2,
                 p_attribute:float|list|np.ndarray = 0.6,
                 attribute_config:list[dict] = list(),
                 node_embedding_dim:int = 16,
                 node_embeddings:Literal["one-hot", "random"] = "one-hot",
                 reflexive:bool = True, seed:int|None = None) -> None:
        """
        :param num_nodes: total number of nodes in the graph
        :param num_relations: total number of unique relations in the graph
        :param num_attributes: total number of attributes in the graph
        :param reflexive: whether nodes can refer to themselves
        :param p_relation: the probability for each node of being connected via
            a relation. If num_relations > 1 then this is expected to be a list
            with as many values as there are relations.
        :param p_attribute: the probability for each node of having one or more
            attributes. If num_attributes > 1 then this is expected to be a list
            with as many values as there are attributes.
        :param attribute_config: list of dictionaries, one per feature,
            with parameters specifying how to generate the values.
            Necessary keys are 'mode', 'embedding_dim', 'distribution',
            and 'p'. If the list is empty, then num_attributes configurations
            are generated randomly.
        :param node_embeddings: use one-hot encoded or random embeddings
        :param node_embedding_dim: size (width) of embeddings if random
        :param seed: seed for random number generator
        :returns: None
        """
        
        super().__init__(num_nodes = num_nodes, num_relations = num_relations,
                         p_relation = p_relation, reflexive = reflexive, 
                         node_labels = True, edge_labels = True,
                         seed = seed)

        # generate node embedding matrix E
        self.E = mkNodeEmbeddings(num_nodes = num_nodes,
                                  mode = node_embeddings,
                                  embedding_dim = node_embedding_dim,
                                  rng = self._rng)

        # generate feature embedding matrix F
        if len(attribute_config) <= 0:  # randomized attributes
            p_attribute: list[float] = [p_attribute] * num_attributes
            attribute_config = list()
            for p in p_attribute:
                mode = self._rng.choice(["natural", "real"],
                                        p = [0.1, 0.9])
                dist = self._rng.choice(["random", "normal",
                                         "exponential", "poisson",
                                         "uniform"], p = [0.4, 0.3, 0.1,
                                                          0.1, 0.1])
                edim = self._rng.choice([1, 2, 4, 8, 16, 32, 64, 128, 256])

                attribute_config.append({'p': p,
                                         "distribution": dist,
                                         "mode": mode,
                                         "embedding_dim": edim})

        self.attributes = attribute_config
        self.F, self.F_idxptr =\
                mkFeatureEmbeddings(num_nodes = num_nodes,
                                    features = self.attributes,
                                    rng = self._rng)
    
    def node_embeddings(self) -> np.ndarray:
        """ Return the node embeddings matrix
        
        :returns: a Numpy array
        """
        return self.E

    
    def freature_embeddings(self) -> np.ndarray:
        """ Return the node feature embeddings matrix
        
        :returns: a Numpy array
        """
        return self.F
