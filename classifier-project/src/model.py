class Classifier(nn.Module):
    """Classifier network with dropout."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        
        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        """Forward pass through the network, returns the output logits."""
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        x = relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout after activation
        x = relu(self.fc2(x))
        x = self.dropout(x)  # Apply dropout after activation
        x = relu(self.fc3(x))
        x = self.dropout(x)  # Apply dropout after activation
        return log_softmax(self.fc4(x), dim=1)