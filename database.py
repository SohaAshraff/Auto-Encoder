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
