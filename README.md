# Developer Tools for Graph Neural Networks

A simple Python library to aid the developers of graph neural networks by offering the generation of random vectorized graphs for the quick testing and debugging of methods. Graphs can be directed or undirected, have labeled vertices and edges, have one or more edge types (relations), be reflexive, as well as have various other characteristics. Convenience functions facilitate the creation of vertex and/or feature embedding matrices, as well as the corresponding adjacency, degree, incidence, and Laplacian matrices.

## Install

Use `git` and `pip` to clone this repository and to install the library, respectively.

```bash
$ git clone https://gitlab.com/wxwilcke/gnn_devtools.git
$ cd gnn_devtools/
$ pip install .
```

## Examples

```Python
>>> import gnn_devtools

>>> g = gnn_devtools.UndirectedGraph(8)

>>> g.A

array([[1, 0, 0, 0, 0, 0, 0, 0],
       [0, 1, 0, 1, 0, 1, 0, 0],
       [0, 0, 1, 1, 1, 0, 0, 0],
       [0, 1, 1, 1, 0, 0, 1, 0],
       [0, 0, 1, 0, 1, 0, 0, 0],
       [0, 1, 0, 0, 0, 1, 0, 0],
       [0, 0, 0, 1, 0, 0, 1, 1],
       [0, 0, 0, 0, 0, 0, 1, 1]], dtype=int8)

>>> g.degree()

array([[2, 0, 0, 0, 0, 0, 0, 0],
       [0, 4, 0, 0, 0, 0, 0, 0],
       [0, 0, 4, 0, 0, 0, 0, 0],
       [0, 0, 0, 5, 0, 0, 0, 0],
       [0, 0, 0, 0, 3, 0, 0, 0],
       [0, 0, 0, 0, 0, 3, 0, 0],
       [0, 0, 0, 0, 0, 0, 4, 0],
       [0, 0, 0, 0, 0, 0, 0, 3]])

>>> g.incidence()

array([[2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 0, 0, 1, 0, 2, 1, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0],
       [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 1, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2]], dtype=int8)

>>> g.laplacian(normalized=True) 

array([[ 0.5       ,  0.        ,  0.        ,  0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.75      ,  0.        , -0.2236068 ,  0.        , -0.28867513,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.75      , -0.2236068 , -0.28867513,  0.        ,  0.        ,  0.        ],
       [ 0.        , -0.2236068 , -0.2236068 ,  0.8       ,  0.        ,  0.        , -0.2236068 ,  0.        ],
       [ 0.        ,  0.        , -0.28867513,  0.        ,  0.66666667,  0.        ,  0.        ,  0.        ],
       [ 0.        , -0.28867513,  0.        ,  0.        ,  0.        ,  0.66666667,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        , -0.2236068 ,  0.        ,  0.        ,  0.75      , -0.28867513],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,  0.        , -0.28867513,  0.66666667]])
```

```Python
>>> kg = gnn_devtools.KnowledgeGraph(num_nodes=8, num_relations=3)

>>> kg.A

array([[[1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 1]],

       [[1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 1, 0, 1]],

       [[1, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 1, 1]]], dtype=int8)

>>> kg.degree(mode="out", collapsed=True)

array([[ 6,  0,  0,  0,  0,  0,  0,  0],
       [ 0,  6,  0,  0,  0,  0,  0,  0],
       [ 0,  0,  9,  0,  0,  0,  0,  0],
       [ 0,  0,  0, 10,  0,  0,  0,  0],
       [ 0,  0,  0,  0,  6,  0,  0,  0],
       [ 0,  0,  0,  0,  0,  6,  0,  0],
       [ 0,  0,  0,  0,  0,  0,  7,  0],
       [ 0,  0,  0,  0,  0,  0,  0,  9]])

>>> kg.n2i  # vertex label to index map

{'N890fc31a1c1b642fea17bd74c851e35a': 0,
 'Nfdae5c7702bb10bff206e5b82104dbde': 1,
 'N75e7699c8bf25e172ccda492647930cf': 2,
 'N42375df55a8b4176a17b5e9e227ac1c0': 3,
 'N9b429ac3654135f407dd6c533bb6019f': 4,
 'Na4a445078c6170145f97a00d4b581132': 5,
 'N28f7539bc851e034e6ff7ec6cce8c398': 6,
 'Ne06fb017daa501c396dc460083dd5161': 7}

>>> kg.E  # vertex embeddings

array([[1, 0, 0, 0, 0, 0, 0, 0],
       [0, 1, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 0, 0, 0, 0, 0],
       [0, 0, 0, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 0, 0, 0],
       [0, 0, 0, 0, 0, 1, 0, 0],
       [0, 0, 0, 0, 0, 0, 1, 0],
       [0, 0, 0, 0, 0, 0, 0, 1]], dtype=int8)

>>> kg.F  # feature embeddings

array([[ 0.        ,  0.42964028, -1.96411919, ...,  1.24073785, -0.85354991, -0.50456663],
       [ 0.        ,  0.34339228,  0.66778396, ...,  0.24497085,  1.03503385,  0.61037809],
       [ 2.85866612,  0.1549609 ,  0.78601422, ...,  0.0187412 ,  1.2640499 ,  0.83004195],
       ...,
       [ 0.        , -0.25434897, -1.24816737, ...,  0.32131674,  0.97800948, -0.80860831],
       [ 0.        , -0.13468174,  0.31076248, ...,  0.94036713, -0.31705597, -0.14261492],
       [ 0.        ,  0.        ,  0.        , ...,  0.        ,  0.        ,  0.        ]])

```
