import torch
from model import AutoEncoder
from database import load_mnist
import matplotlib.pyplot as plt

model = AutoEncoder()
model.load_state_dict(torch.load("autoencoder.pth"))  
model.eval()

_, test_loader = load_mnist(batch_size=1)

for x, _ in test_loader:
    output = model(x)

    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(x[0][0], cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title("Reconstructed")
    plt.imshow(output[0][0].detach(), cmap='gray')

    plt.show()
    break  