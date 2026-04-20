import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size=3072, hidden_sizes=[512, 256], num_classes=6, dropout=0.3):
        super().__init__()
        layers = []
        in_size = input_size
        for h in hidden_sizes:
            layers += [
                nn.Linear(in_size, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_size = h
        layers.append(nn.Linear(in_size, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        return self.net(x)