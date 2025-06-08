import torch
from torchvision import transforms
import IPython.display as display
import matplotlib.pyplot as plt
path = "corruptmnist_v1"

test_dataloaders = torch.load(f"{path}/test_images.pt")
test_dataloaders
print(test_dataloaders.shape)
train_dataloaders = torch.load(f"{path}/train_images_0.pt")
train_dataloaders
print(train_dataloaders.shape)
# def corrupt_mnist(torch.utils.data.Dataset) -> tuple[torch.Tensor, torch.Tensor]:
#     """Return train and test dataloaders for corrupt MNIST."""
#     # exchange with the corrupted mnist dataset
#     test_dataloaders = torch.load(f"{path}/test_images.pt")
#     train_dataloaders = torch.load(f"{path}/train_images_0.pt")
#     transform = 
#     train = torch.randn(50000, 784)
#     test = torch.randn(10000, 784)
#     return train, test
# def show_image_from_mnist


