import torch
from gsea.model import GaborSemanticEvidence
from gsea.losses import training_loss

def test_end_to_end():
    torch.manual_seed(5)
    cfg=dict(num_concepts=4,num_classes=4,top_m=3,proj_dim=16,attention_hidden=8,
             num_orientations=2,num_frequencies=2,gabor_kernel_size=7,k_neighbors=2,
             spatial_tau=.2,morph_tau=1.,propagation_steps=1,focal_gamma=2.,graph_smoothness_lambda=.01)
    model=GaborSemanticEvidence(8,12,torch.randn(4,10),cfg)
    slide={'resnet':torch.randn(5,8),'uni':torch.randn(5,12),'coords':torch.rand(5,2)*1000,
           'patches':torch.rand(5,3,24,24),'label':torch.tensor(2)}
    r=model(slide);assert r['logits'].shape==(4,)
    assert r['adjacency'].shape==(3,3)
    assert torch.allclose(r['adjacency'].sum(-1),torch.ones(3),atol=1e-5)
    loss,_=training_loss(r,slide['label'],cfg,'cpu');loss.backward()
    assert model.gabor.theta.grad is not None
    assert model.visual_to_text[0].weight.grad is not None
