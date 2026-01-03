"""Dataset wrapper for binary clean-vs-corrupted classification.
This module defines a PyTorch `Dataset` that:
- Loads CIFAR-10 images (as PIL images)
- With probability `p_corrupt`, applies ONE random corruption (noise/blur/jpeg/occlusion)
- Outputs a tuple (x, y) where:
    x = normalized image tensor, shape [3, H, W]
    y = 0 for clean, 1 for corrupted

Key idea: corruptions are applied *on the fly* in __getitem__, so the same base image
can be corrupted differently across epochs.
"""
import random
from torchvision import datasets, transforms
from torch.utils.data import Dataset

# Our local corruption function (defined in src/corruptions.py)
from .corruptions import corrupt_randomly


class CleanVsCorruptDataset(Dataset):
    """Binary dataset: clean (0) vs corrupted (1) images."""

    def __init__(self, root: str = "./data", train: bool = True, p_corrupt: float = 0.5):
        """Initialize the dataset.

        Args:
            root: Where to store/download CIFAR-10.
            train: If True, use CIFAR-10 train split; else test split.
            p_corrupt: Probability that an image is corrupted on each access.
        """
        # Load CIFAR-10. Each item from self.base is: (PIL_image, class_label)
        self.base = datasets.CIFAR10(root=root, train=train, download=True)

        # Probability of applying a corruption when fetching an item
        self.p_corrupt = p_corrupt

        # Convert PIL image -> torch tensor in range [0, 1], shape [C, H, W]
        self.to_tensor = transforms.ToTensor()

        # Normalize using common CIFAR-10 mean/std (helps model training converge)
        self.normalize = transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        )

    def __len__(self) -> int:
        """Return number of images in the dataset split."""
        return len(self.base)

    def __getitem__(self, idx: int):
        """Get one training example.

        Returns:
            x: torch.FloatTensor, normalized, shape [3, 32, 32]
            y: int, 0 = clean, 1 = corrupted
        """
        # Fetch a CIFAR-10 example
        pil_img, _ = self.base[idx]  # '_' is the original CIFAR-10 class label (0-9), unused here

        # Decide whether to corrupt this image
        if random.random() < self.p_corrupt:
            # Apply exactly one random corruption (noise/blur/jpeg/occlusion)
            pil_img = corrupt_randomly(pil_img)
            y = 1  # label: corrupted
        else:
            y = 0  # label: clean

        # Convert to tensor and normalize for the model
        x = self.normalize(self.to_tensor(pil_img))

        # Return (features, label)
        return x, y