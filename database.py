from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_mnist(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_data = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
import torch

def preprocess_images(images):
    # Normalize between 0 and 1 or any other preprocessing logic
    images = images.view(images.size(0), -1)  # flatten
    images = images / 255.0 if images.max() > 1 else images
    return images

def denormalize_images(images):
    # convert back to image shape for visualization
    return images.view(-1, 1, 28, 28)
