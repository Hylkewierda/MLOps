import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_shape, hidden_units, num_classes):
        super().__init__()

        # Calculate flattened input dimension
        input_dim = 1
        for d in input_shape:
            input_dim *= d

        layers = []
        prev_dim = input_dim

        for h in hidden_units:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h

        layers.append(nn.Linear(prev_dim, num_classes))

        # Register layers properly
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)

        # IMPORTANT: return output
        return self.net(x)

