import torch
from torch.nn import functional as F

def focal_loss(logits,target,gamma=2.0,class_weights=None):
    log_probs=F.log_softmax(logits[None],-1)
    p=log_probs.exp()
    ce=F.nll_loss(log_probs,target.reshape(1),weight=class_weights)
    pt=p[0,target]
    return (1-pt).pow(gamma)*ce

def smoothness_loss(evidence,adjacency):
    # Directed row stochastic smoothness energy. No symmetric-Laplacian assumption.
    return (adjacency * torch.cdist(evidence,evidence).square()).sum()/adjacency.sum().clamp_min(1e-8)

def training_loss(result,target,cfg,device):
    weights=cfg.get('class_weights')
    if weights is not None: weights=torch.tensor(weights,device=device,dtype=torch.float32)
    # This is standard focal classification, NOT ordinal focal loss.
    ce=focal_loss(result['logits'],target,cfg['focal_gamma'],weights)
    smooth=smoothness_loss(result['propagated'],result['adjacency'])
    return ce+cfg['graph_smoothness_lambda']*smooth,{'focal':float(ce.detach()),'smooth':float(smooth.detach())}
