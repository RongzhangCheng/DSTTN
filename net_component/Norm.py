import torch
import torch.nn as nn


class SNorm(nn.Module):
    def __init__(self, channels):
        super(SNorm, self).__init__()
        self.beta = nn.Parameter(torch.zeros(channels))
        self.gamma = nn.Parameter(torch.ones(channels))

    def forward(self, x):
        x_norm = (x - x.mean(2, keepdims=True)) / (x.var(2, keepdims=True, unbiased=True) + 0.00001) ** 0.5
        out = x_norm + self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)
        return out


class TNorm(nn.Module):
    def __init__(self, num_nodes, channels, track_running_stats=True, momentum=0.1):
        super(TNorm, self).__init__()
        self.track_running_stats = track_running_stats
        self.beta = nn.Parameter(torch.zeros(1, channels, num_nodes, 1))
        self.gamma = nn.Parameter(torch.ones(1, channels, num_nodes, 1))
        self.register_buffer('running_mean', torch.zeros(1, channels, num_nodes, 1))
        self.register_buffer('running_var', torch.ones(1, channels, num_nodes, 1))
        self.momentum = momentum

    def forward(self, x):
        if self.track_running_stats:
            mean = x.mean((0, 3), keepdims=True)
            var = x.var((0, 3), keepdims=True, unbiased=False)
            if self.training:
                n = x.shape[3] * x.shape[0]
                with torch.no_grad():
                    self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
                    self.running_var = self.momentum * var * n / (n - 1) + (1 - self.momentum) * self.running_var
            else:
                mean = self.running_mean
                var = self.running_var
        else:
            mean = x.mean((3), keepdims=True)
            var = x.var((3), keepdims=True, unbiased=True)
        x_norm = (x - mean) / (var + 0.00001) ** 0.5
        out = x_norm + self.gamma + self.beta
        return out


class STNorm(nn.Module):
    def __init__(self, hidden_units, num_nodes) -> None:
        super().__init__()
        self.spatial_norm = SNorm(channels=hidden_units)
        self.temporal_norm = TNorm(num_nodes=num_nodes, channels=hidden_units)

    def forward(self, x):
        h = x.permute(1, 0, 2).unsqueeze(1)
        hs = self.spatial_norm(h)
        ht = self.temporal_norm(h)
        
        hs = hs.mean(dim=1, keepdim=True)
        ht = ht.mean(dim=1, keepdim=True)

        h = h.permute(0, 3, 2, 1)
        hs = hs.permute(0, 3, 2, 1)
        ht = ht.permute(0, 3, 2, 1)

        return h, hs, ht
