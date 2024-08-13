from torch import nn


class PositionWiseFeedForwardNetwork(nn.Module):

    def __init__(self, dim_model, dim_ff):
        super().__init__()
        self.layer1 = nn.Linear(dim_model, dim_ff)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(dim_ff, dim_model)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)

        return x