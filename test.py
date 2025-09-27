import torch
from model import AutoEncoder
from database import load_mnist
import matplotlib.pyplot as plt

model = AutoEncoder()
model.load_state_dict(torch.load("autoencoder.pth"))
model.eval()

_, test_loader = load_mnist(batch_size=8)

for x, _ in test_loader:
    output = model(x)

    n = x.size(0)
    
    plt.figure(figsize=(16, 4))
    
    for i in range(n):
        plt.subplot(2, n, i+1)
        plt.title("Original")
        plt.imshow(x[i][0], cmap='gray')
        plt.axis('off')
        
        plt.subplot(2, n, n+i+1)
        plt.title("Reconstructed")
        plt.imshow(output[i][0].detach(), cmap='gray')
        plt.axis('off')
    
    plt.show()
    break  