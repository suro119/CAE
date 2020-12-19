import numpy as np
import torch
import torch.nn as nn

X = torch.distributions.normal.Normal(4, 1)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = nn.Parameter(torch.Tensor([0]))
        self.var = nn.Parameter(torch.Tensor([1]))

    def forward(self, x):
        Ndist = torch.distributions.normal.Normal(self.mean, self.var)
        out = -Ndist.log_prob(x)
        return out


mdl = Model().to(0)
optim = torch.optim.Adam(mdl.parameters(), lr=1e-2)

for _ in range(50000):
    nlog_prob = mdl(X.sample().to(0))
    obj = nlog_prob.mean()
    optim.zero_grad()
    obj.backward()
    optim.step()

print(mdl.mean, mdl.var)
