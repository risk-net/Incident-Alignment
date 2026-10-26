"""Single public entrypoint. All nodes retained, including isolates."""
from .contracts import ClusteringConfig
from .edges import normalize_nodes,undirected_scores
from .dense_backend import cluster_dense
from .sparse_backend import build_sparse_complete_link_clusters

def cluster_events(nodes, directed_scores, config:ClusteringConfig):
    nodes=normalize_nodes(nodes)
    und=undirected_scores(nodes,directed_scores,config.edge_rule)
    if config.backend=="dense":
        result=cluster_dense(nodes,und,config.threshold)
    else:
        # Sparse backend has its own symmetrization; pass canonical symmetric scores
        symmetric={(u,v):s for (u,v),s in und.items()}
        symmetric.update({(v,u):s for (u,v),s in und.items()})
        result=build_sparse_complete_link_clusters(nodes,symmetric,"mutual",config.threshold)
    flat=[n for c in result for n in c]
    if len(flat)!=len(nodes) or set(flat)!=set(nodes) or len(flat)!=len(set(flat)):
        raise RuntimeError("clustering violates domain partition contract")
    return result
