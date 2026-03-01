import numpy as np
from sklearn.neighbors import NearestNeighbors

def _knn_edges(X: np.ndarray, k: int = 10):
    k = int(min(max(k, 2), len(X)))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(X)
    d, idx = nn.kneighbors(X)

    edges = []
    weights = []
    for i in range(len(X)):
        for jpos, j in enumerate(idx[i]):
            if i == j:
                continue
            w = 1.0 - float(d[i][jpos])
            if w < 0:
                w = 0.0
            edges.append((int(i), int(j)))
            weights.append(float(w))
    return edges, weights

def leiden_membership(X: np.ndarray, k: int = 10, resolution: float = 1.0):
    try:
        import igraph as ig
        import leidenalg
    except Exception as e:
        raise RuntimeError("Leiden requires igraph and leidenalg. Install: pip install igraph leidenalg") from e

    edges, weights = _knn_edges(X, k=k)
    g = ig.Graph(n=len(X), edges=edges, directed=False)
    g.es["weight"] = weights
    part = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=float(resolution),
    )
    return list(part.membership)
