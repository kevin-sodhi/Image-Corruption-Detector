import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import our custom dataset that returns (image_tensor, label)
# label: 0 = clean, 1 = corrupted
from .data import CleanVsCorruptDataset

class SmallCNN(nn.Module):
    """
    A simple CNN for 32x32 RGB images (CIFAR-10 size).
    Outputs 2 logits: [clean_logit, corrupt_logit]
    """
    def __init__(self):
        super().__init__()

        # Feature extractor + classifier
        self.net = nn.Sequential(
            # Input: [B, 3, 32, 32] -> [B, 16, 32, 32]
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            # Downsample: -> [B, 16, 16, 16]
            nn.MaxPool2d(2),

            # -> [B, 32, 16, 16]
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            # Downsample: -> [B, 32, 8, 8]
            nn.MaxPool2d(2),

            # Flatten [B, 32, 8, 8] -> [B, 32*8*8]
            nn.Flatten(),

            # Fully-connected layers
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),

            # Final layer: 2 classes (clean vs corrupt)
            nn.Linear(64, 2),
        )

    def forward(self, x):
        """Forward pass: returns logits of shape [B, 2]."""
        return self.net(x)


def get_device():
    """
    Pick the best available device:
    - 'cuda' for NVIDIA GPU
    - 'mps' for Apple Silicon GPU (Metal Performance Shaders)
    - 'cpu' otherwise
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main(epochs=2, batch_size=128, lr=1e-3):
    """
    Train a clean-vs-corrupt classifier.

    Args:
        epochs: number of passes over the training set
        batch_size: batch size for training/validation
        lr: learning rate
    """
    # place to save the model
    os.makedirs("models", exist_ok=True)

    # Choose device (cpu/cuda/mps)
    device = get_device()
    print("Device:", device)

    # Create datasets
    # p_corrupt=0.5 means ~half the time we corrupt the image (label=1)
    train_ds = CleanVsCorruptDataset(train=True, p_corrupt=0.5)
    val_ds = CleanVsCorruptDataset(train=False, p_corrupt=0.5)

    # Wrap datasets in DataLoaders for batching
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # Create model and move it to chosen device
    model = SmallCNN().to(device)

    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Loss for multi-class classification (2 classes here)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        # Iterate over training batches with a progress bar
        for x, y in tqdm(train_loader, desc=f"Epoch {ep}/{epochs}"):
            # Move data to device
            x, y = x.to(device), y.to(device)

            # Reset gradients
            opt.zero_grad()

            # Forward pass -> logits
            logits = model(x)

            # Compute loss
            loss = loss_fn(logits, y)

            # Backprop + update weights
            loss.backward()
            opt.step()

            total_loss += loss.item()

        # Validation loop (no gradients)
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)

                # Predicted class = argmax of logits
                pred = model(x).argmax(dim=1)

                # Count correct predictions
                correct += (pred == y).sum().item()
                total += y.numel()

        val_acc = correct / total

        # Average training loss over batches
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {ep}: loss={avg_loss:.4f} val_acc={val_acc:.4f}")

    # Save trained weights
    torch.save(model.state_dict(), "models/clean_vs_corrupt_cnn.pt")
    print("Saved -> models/clean_vs_corrupt_cnn.pt")


if __name__ == "__main__":
    main()
