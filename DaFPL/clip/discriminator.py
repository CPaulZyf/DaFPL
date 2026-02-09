import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainDiscriminator(nn.Module):
    """
    Simple MLP domain discriminator for prompt representations.
    Input:  [B, in_dim]
    Output: [B, K] logits, K = num_domains
    """
    def __init__(
        self,
        in_dim: int,
        num_domains: int,
        hidden_dims=[32],
        dropout: float = 0.1,
        use_bn: bool = True,
        temperature: float = 1.0,
    ):
        super().__init__()
        assert temperature > 0
        self.temperature = float(temperature)

        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h

        self.net = nn.Sequential(*layers) if layers else nn.Identity()
        self.head = nn.Linear(prev, num_domains)

        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return self.head(z) / self.temperature
