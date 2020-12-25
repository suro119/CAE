import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from .networks import init_net


def get_entropy_model(model, scale, gpu_id=None):
    if model == 'gsm':
        entropy_model = GaussianScaleMixture(scale, gpu_id)
    else:
        raise NotImplementedError('Entropy model name \'{}\' not implemented'.format(model))
    return init_net(entropy_model, gpu_id=gpu_id)


class GaussianScaleMixture(nn.Module):
    def __init__(self, scale, gpu_id):
        super().__init__()
        self.weights = nn.Parameter(torch.rand(96,6)).to(gpu_id)
        stds = torch.rand(96,6,1).to(gpu_id)
        stds += scale
        self.stds = nn.Parameter(stds)

        self.means = torch.zeros(96,6,1).to(gpu_id)
        self.eps = torch.tensor([1e-4]).to(gpu_id)
        #self.zeros = torch.tensor([0]).to(gpu_id)

    def forward(self, x):
        stds = F.relu(self.stds) + self.eps # Positive std only
        mix = D.Categorical(self.weights)
        comp = D.Independent(Normal_(self.means, stds), 1)
        gsm = D.MixtureSameFamily(mix, comp)
        x = x.view(x.size(0), x.size(1), -1)    
        #log_probs = gsm.log_prob(x)
        log_probs = gsm.log_prob(x)
        out = torch.mean(log_probs)

        if out > 0:
            print(x[:,-1,:])
            print(self.weights[-1,:])
            print(self.stds[-1,:])
            print(log_probs[:,-1])
            print(out)
            raise ValueError('NaN Detected\n')

        return out


class Normal_(D.Normal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def log_prob(self, x):
        # out = super().log_prob(x+0.5) + torch.log(torch.cosh(0.5*x)) + 0.5*x + torch.log(torch.zeros_like(x)+2)
        # if torch.isnan(out).any():
        #     print(super().log_prob(x+0.5), torch.log(torch.cosh(0.5*x)), 0.5*x, torch.log(torch.zeros_like(x)+2))
        #     print(x)
        #     raise ValueError('NaN Detected\n')
        out = super().log_prob(x+0.5) + super().log_prob(x-0.5) + 2*super().log_prob(x) - torch.log(torch.zeros_like(x)+4)
        return out


# class Independent_(D.Independent):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def cdf(self, x):  # B, C, H*W
#         cdf = self.base_dist.cdf(x)  # B, C, 6, H*W
#         if torch.where(cdf == 0.0, True, False).any() or torch.where(cdf == 1.0, True, False).any():
#             print(cdf[cdf==1].size())
#             print(cdf[cdf==0].size())
#             raise ValueError('cdf returns 0 or 1')

#         return cdf


# # Impelemnts 'cdf' function for GSMs
# class MixtureSameFamily_(D.MixtureSameFamily):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
    
#     def relaxed_log_prob(self, x):
#         upper = x + 0.5
#         lower = x - 0.5
#         # self._pad operation in MixtureSameFamily for an Independent Gaussian
#         upper = self._pad(upper)
#         lower = self._pad(lower)
#         cdf_upper = self.component_distribution.cdf(upper)  # B, C, 6, H*W
#         cdf_lower = self.component_distribution.cdf(lower)  # B, C, 6, H*W
#         mix_prob = self.mixture_distribution.probs  # C, 6
#         mix_prob = mix_prob.unsqueeze(-1)  # C, 6, 1
#         if torch.where(mix_prob == 0.0, True, False).any():
#             print(mix_prob)
#             raise ValueError('mix prob 0')
#         cdf_upper = torch.sum(cdf_upper * mix_prob, dim=2)  # B, C, H*W
#         cdf_lower = torch.sum(cdf_lower * mix_prob, dim=2)  # B, C, H*W

#         probs = cdf_upper - cdf_lower  # B, C, H*W  # does not work due to numerical issues
#         if torch.where(probs == 0.0, True, False).any():
#             print(probs)
#             raise ValueError('cdf subtract 0')
#         probs = torch.prod(probs, dim=-1)  # B, C
#         probs = torch.prod(probs, dim=-1)  # B
#         log_probs = torch.log(probs)
        
#         return log_probs