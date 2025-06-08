from torch import nn


class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self,input , hidden_layer, dropouts) -> None:
        super().__init__()
        nn.Sequential(
        
            nn.Linear(input, hidden_layer[0]),
            nn.ReLU(),
            nn.Dropout(dropouts[0]),
            nn.Linear(),
        )
    def forward(self, x):
        """Forward pass through the model."""
        for layer in self.hidden_layers:
            x = layer(x)
        
        return x   
