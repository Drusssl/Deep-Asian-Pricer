import torch
import torch.nn as nn

class PricingNet(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=256):
        super(PricingNet, self).__init__()
        # 4-Layer MLP with SiLU (Swish) activation for smooth derivatives
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1) # Output: Option Price
        )
        
    def forward(self, x):
        return self.net(x)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)