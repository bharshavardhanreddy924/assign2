import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from skimage.util import view_as_windows
import os

# DnCNN Model Definition
class DnCNN(nn.Module):
    def __init__(self, channels=3, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, 
                                kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, 
                                    kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, 
                                kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        noise = self.dncnn(x)
        return x - noise

def read_raw_image(filepath, width=1920, height=1280):
    """
    Read 12-bit RAW image in GRBG Bayer pattern
    """
    try:
        with open(filepath, 'rb') as f:
            raw_data = np.fromfile(f, dtype=np.uint16)
            
        if len(raw_data) != width * height:
            raise ValueError(f"Expected {width*height} pixels, got {len(raw_data)}")
            
        # Reshape and mask to 12 bits
        raw_image = raw_data.reshape((height, width))
        raw_image = raw_image & 0x0FFF
        raw_image = raw_image.astype(np.uint16)
        
        # Scale to full 16-bit range
        raw_image = (raw_image << 4).astype(np.uint16)
        return raw_image
    except Exception as e:
        print(f"Error loading RAW image: {e}")
        return None

def demosaic_image(bayer_image):
    """
    Edge-aware demosaicing for GRBG Bayer pattern
    """
    try:
        return cv2.cvtColor(bayer_image, cv2.COLOR_BAYER_GR2RGB)
    except Exception as e:
        print(f"Demosaicing error: {e}")
        return None

def preprocess_image(image):
    """
    Preprocess image for DnCNN training
    """
    # Convert to float and normalize
    image = image.astype(np.float32) / 255.0
    return image

def generate_noisy_image(clean_image, noise_std=0.05):
    """
    Generate synthetic noisy image
    """
    # Add Gaussian noise
    noise = np.random.normal(0, noise_std, clean_image.shape).astype(np.float32)
    noisy_image = np.clip(clean_image + noise, 0, 1)
    return noisy_image

def extract_patches(image, patch_size=40, stride=10):
    """
    Extract patches from image
    """
    # Ensure image is in [0,1] range
    if image.max() > 1:
        image = image.astype(np.float32) / 255.0
    
    # Transpose to match PyTorch channel order
    image = image.transpose(2, 0, 1)
    
    # Extract patches
    patches = view_as_windows(image, (image.shape[0], patch_size, patch_size), step=(image.shape[0], stride, stride))
    
    # Reshape patches
    patches = patches.squeeze(0).reshape(-1, image.shape[0], patch_size, patch_size)
    
    return patches

def train_dncnn(clean_patches, noisy_patches, epochs=1, lr=0.01, batch_size=16):
    # Convert to PyTorch tensors and move to GPU (if available)
    clean_patches = torch.from_numpy(clean_patches).float()
    noisy_patches = torch.from_numpy(noisy_patches).float()
    
    # Initialize model
    model = DnCNN(channels=3)
    model = model.cuda() if torch.cuda.is_available() else model
    clean_patches = clean_patches.cuda() if torch.cuda.is_available() else clean_patches
    noisy_patches = noisy_patches.cuda() if torch.cuda.is_available() else noisy_patches

    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        # Loop over mini-batches
        for i in range(0, len(clean_patches), batch_size):
            batch_size_current = min(batch_size, len(clean_patches) - i)
            batch_clean = clean_patches[i:i+batch_size_current]
            batch_noisy = noisy_patches[i:i+batch_size_current]

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            noise_pred = model(batch_noisy)

            # Compute loss
            loss = criterion(noise_pred, batch_noisy - batch_clean)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Print progress
            if i % (batch_size * 10) == 0:
                print(f'Epoch [{epoch}/{epochs}], Step [{i}/{len(clean_patches)}], Loss: {loss.item():.4f}')

    return model  # Return the trained model

def create_dncnn_weights(image_path):
    """
    Main function to create DnCNN weights
    """
    # Read RAW image
    raw_image = read_raw_image(image_path)
    if raw_image is None:
        print("Failed to read image")
        return
    
    # Demosaic
    clean_image = demosaic_image(raw_image)
    if clean_image is None:
        print("Demosaicing failed")
        return
    
    # Preprocess
    clean_image = preprocess_image(clean_image)
    
    # Generate noisy image
    noisy_image = generate_noisy_image(clean_image)
    
    # Extract patches
    clean_patches = extract_patches(clean_image)
    noisy_patches = extract_patches(noisy_image)
    
    # Train model and save weights
    model = train_dncnn(clean_patches, noisy_patches)
    
    # Specify the path to save the weights
    save_path = r"C:\Users\borut\Desktop\industry assign\dncnn\dncnn_custom.pth"
    
    # Save model weights
    torch.save(model.state_dict(), save_path)
    print(f"Weights saved successfully at {save_path}!")

# Example usage
if __name__ == "__main__":
    # Specify the path to your RAW image
    image_path = r"C:\Users\borut\Desktop\industry assign\dncnn\eSFR_1920x1280_12b_GRGB_6500K_60Lux.raw"
    create_dncnn_weights(image_path)

# Example usage
if __name__ == "__main__":
    # Specify the path to your RAW image
    image_path = r"C:\Users\borut\Desktop\industry assign\dncnn\eSFR_1920x1280_12b_GRGB_6500K_60Lux.raw"
    create_dncnn_weights(image_path)


# Example usage
if __name__ == "__main__":
    # Specify the path to your RAW image
    image_path = r"C:\Users\borut\Desktop\industry assign\dncnn\eSFR_1920x1280_12b_GRGB_6500K_60Lux.raw"
    create_dncnn_weights(image_path)