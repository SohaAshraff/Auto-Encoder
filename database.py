import torch

def preprocess_images(images):
    # Normalize between 0 and 1 or any other preprocessing logic
    images = images.view(images.size(0), -1)  # flatten
    images = images / 255.0 if images.max() > 1 else images
    return images

def denormalize_images(images):
    # convert back to image shape for visualization
    return images.view(-1, 1, 28, 28)
