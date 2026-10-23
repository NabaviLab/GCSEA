import torch
from torch import nn
from torch.nn import functional as F
from .gabor import LearnableGabor
from .graph import make_adjacency

class GaborSemanticEvidence(nn.Module):
    """Inputs: frozen ResNet/UNI embeddings, raw patches, original WSI coordinates.

    CONCH concept vectors must be produced by the REAL frozen CONCH text encoder.
    A learnable projection aligns ResNet features with this vector space under grade supervision.
    Grade labels alone do not guarantee clinically valid semantic alignment.
    """
    def __init__(self, resnet_dim, uni_dim, concept_embeddings, cfg):
        super().__init__()
        if concept_embeddings.ndim!=2: raise ValueError('Concept embeddings need [K,D]')
        k,tdim=concept_embeddings.shape
        if k!=cfg['num_concepts']: raise ValueError('Configured number of concepts does not match embeddings')
        self.register_buffer('concepts', F.normalize(concept_embeddings.float(),dim=-1))
        self.selector=nn.Linear(uni_dim,1)
        self.visual_to_text=nn.Sequential(nn.Linear(resnet_dim,cfg['proj_dim']),nn.ReLU(),nn.Linear(cfg['proj_dim'],tdim))
        self.gabor=LearnableGabor(cfg['num_orientations'],cfg['num_frequencies'],cfg['gabor_kernel_size'])
        self.attention=nn.Sequential(nn.Linear(k,cfg['attention_hidden']),nn.Tanh(),nn.Linear(cfg['attention_hidden'],1))
        self.classifier=nn.Linear(k,cfg['num_classes'])
        self.cfg=cfg

    def forward(self, slide):
        h=slide['resnet']; u=slide['uni']; coords=slide['coords']; patches=slide['patches']
        if not (len(h)==len(u)==len(coords)==len(patches)): raise ValueError('Patch alignment corrupted')
        importance=torch.sigmoid(self.selector(u).squeeze(-1))
        idx=importance.topk(min(self.cfg['top_m'],len(importance))).indices
        weighted_h=h[idx]*importance[idx,None]
        projected=F.normalize(self.visual_to_text(weighted_h),dim=-1)
        evidence=projected@self.concepts.T  # [M,K] concept evidence
        morphology=self.gabor(patches[idx])
        adjacency=make_adjacency(coords[idx],morphology,self.cfg['k_neighbors'],self.cfg['spatial_tau'],self.cfg['morph_tau'])
        propagated=evidence
        for _ in range(self.cfg['propagation_steps']): propagated=adjacency@propagated
        weights=torch.softmax(self.attention(propagated).squeeze(-1),dim=0) # patch weights
        concept_vector=(weights[:,None]*propagated).sum(0)
        logits=self.classifier(concept_vector)
        return {'logits':logits,'evidence':evidence,'propagated':propagated,'adjacency':adjacency,
                'attention':weights,'selected_indices':idx,'importance':importance,'slide_embedding':concept_vector}
