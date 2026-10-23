import math
import torch
from torch import nn
from torch.nn import functional as F

class LearnableGabor(nn.Module):
    """Differentiable real Gabor bank; responses computed on raw selected RGB patches."""
    def __init__(self, orientations=4, frequencies=2, kernel_size=15):
        super().__init__()
        if kernel_size % 2 != 1: raise ValueError('Gabor kernel size must be odd')
        self.orientations, self.frequencies, self.kernel_size = orientations, frequencies, kernel_size
        b = orientations * frequencies
        theta = torch.arange(orientations).repeat_interleave(frequencies).float() * math.pi / orientations
        freq = torch.linspace(0.06, 0.22, frequencies).repeat(orientations)
        self.theta = nn.Parameter(theta)
        self.raw_freq = nn.Parameter(torch.log(torch.expm1(freq)))
        self.raw_sigma = nn.Parameter(torch.full((b,), math.log(math.expm1(3.0))))
        self.raw_gamma = nn.Parameter(torch.full((b,), math.log(math.expm1(0.65))))
        self.phi = nn.Parameter(torch.zeros(b))
        x = torch.arange(kernel_size).float() - kernel_size//2
        yy, xx = torch.meshgrid(x, x, indexing='ij')
        self.register_buffer('xx', xx); self.register_buffer('yy', yy)

    def kernels(self):
        theta=self.theta[:,None,None]
        xp=self.xx * theta.cos() + self.yy * theta.sin()
        yp=-self.xx * theta.sin() + self.yy * theta.cos()
        f=F.softplus(self.raw_freq)[:,None,None].clamp(0.01,0.5)
        sigma=F.softplus(self.raw_sigma)[:,None,None].clamp_min(.1)
        gamma=F.softplus(self.raw_gamma)[:,None,None].clamp_min(.01)
        phase=self.phi[:,None,None]
        kernel=torch.exp(-(xp.square()+gamma.square()*yp.square())/(2*sigma.square())) * torch.cos(2*math.pi*f*xp+phase)
        kernel = kernel - kernel.mean(dim=(-1,-2),keepdim=True)
        kernel = kernel / (kernel.square().sum(dim=(-1,-2),keepdim=True).sqrt()+1e-8)
        return kernel[:,None,:,:]

    def forward(self, patches):
        """patches [M,3,H,W] in [0,1]; output energies [M,O*F]."""
        if patches.ndim != 4 or patches.shape[1]!=3: raise ValueError('Expected RGB patch tensor')
        gray=(patches[:,0:1]*0.2989+patches[:,1:2]*0.587+patches[:,2:3]*0.114)
        response=F.conv2d(gray, self.kernels(), padding=self.kernel_size//2)
        return response.square().mean(dim=(-1,-2))
