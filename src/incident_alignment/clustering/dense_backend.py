"""Per-threshold connected-component SciPy complete linkage. No global tree reuse."""
from collections import defaultdict
import numpy as np
from scipy.cluster.hierarchy import linkage,fcluster
from scipy.spatial.distance import squareform

def cluster_dense(nodes, undirected, threshold):
    parent={x:x for x in nodes}
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    for (u,v),s in undirected.items():
        if s>=threshold:parent[find(u)]=find(v)
    components=defaultdict(list)
    for x in nodes:components[find(x)].append(x)
    result=[]
    for members in components.values():
        members=sorted(members);n=len(members)
        if n==1:result.append(members);continue
        index={x:i for i,x in enumerate(members)}
        mat=np.ones((n,n),dtype=np.float64);np.fill_diagonal(mat,0.)
        for (u,v),s in undirected.items():
            if u in index and v in index:
                i,j=index[u],index[v];mat[i,j]=mat[j,i]=1.-s
        z=linkage(squareform(mat,checks=False),method="complete",optimal_ordering=False)
        labels=fcluster(z,t=1.-threshold+1e-12,criterion="distance")
        groups=defaultdict(list)
        for node,lab in zip(members,labels):groups[int(lab)].append(node)
        result.extend(groups.values())
    return sorted((sorted(c) for c in result),key=lambda c:(c[0],len(c)))
