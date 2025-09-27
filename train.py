import torch
import torch.nn as nn
import torch.optim as optim
from model import AutoEncoder
from database import load_mnist, preprocess_images
def loss_function(recon_x, x, z_mean, z_log_var):
    BCE = nn.functional.mse_loss(recon_x, x.view(-1,1,28,28), reduction='sum')
    KLD = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
    return BCE + KLD
def train_model(num_epochs=10, batch_size=64, lr=1e-3):
    train_loader, _ = load_mnist(batch_size)
    model = AutoEncoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for imgs, _ in train_loader:
            imgs = preprocess_images(imgs)
            outputs,_,_ = model(imgs)
            loss = criterion(outputs, imgs.view(imgs.size(0), 1, 28, 28))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "autoencoder_Fashion.pth")

if __name__ == "__main__":
    train_model()
