import torch
from torch.nn import functional as F

def make_adjacency(coords, morph, k_neighbors=8, spatial_tau=0.2, morph_tau=1.0):
    """Directed spatial kNN graph; row stochastic normalization; self loops guarantee nonzero rows."""
    n=coords.shape[0]
    if n==1: return coords.new_ones((1,1))
    # Coordinates normalized by per-slide spatial extent, preserving relative tissue locality.
    xy=(coords-coords.min(0).values)/(coords.max(0).values-coords.min(0).values).clamp_min(1.)
    dp=torch.cdist(xy,xy).square()
    dg=torch.cdist(F.normalize(morph,dim=-1,eps=1e-8),F.normalize(morph,dim=-1,eps=1e-8)).square()
    k=min(k_neighbors,n-1)
    # Neighbors chosen from spatial distance, not learnable weights.
    neighbor_indices=dp.detach().clone(); neighbor_indices.fill_diagonal_(float('inf'))
    idx=torch.topk(neighbor_indices,k=k,largest=False,dim=-1).indices
    mask=torch.zeros_like(dp).scatter_(1,idx,1.)
    weights=torch.exp(-dp/spatial_tau) * torch.exp(-dg/morph_tau) * mask
    weights=weights+torch.eye(n,device=coords.device,dtype=coords.dtype)
    return weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
