"""Image corruption utilities.
This file defines a few simple *synthetic corruptions* you can apply to a PIL image:
- Gaussian noise
- Gaussian blur
- JPEG compression artifacts
- Black rectangle occlusion
These are useful for creating a "corrupted" version of clean images for training/testing
(e.g., corruption detection, robustness benchmarking, or data augmentation).
"""
import io
import random

import numpy as np
from PIL import Image, ImageFilter


def add_noise(pil_img, sigma_range=(8, 35)):
    """Add Gaussian (normal) noise to an image.
    Args:
        pil_img: Input PIL Image.
        sigma_range: Tuple (min_sigma, max_sigma). A random sigma is sampled each call.
    Returns:
        A new PIL Image with additive Gaussian noise.
    """
    img = np.array(pil_img).astype(np.float32) # Convert PIL -> NumPy float32 so we can do math safely
    sigma = random.uniform(*sigma_range) # Randomly choose the noise strength (standard deviation)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32) # Create a noise matrix the same shape as the image (H, W, C)
    out = np.clip(img + noise, 0, 255).astype(np.uint8)  # Add noise, then clamp to valid pixel range [0, 255]
    # Convert back to PIL
    return Image.fromarray(out)


def add_blur(pil_img, radius_range=(1.0, 3.0)):
    """Apply Gaussian blur.
    Args:
        pil_img: Input PIL Image.
        radius_range: Tuple (min_radius, max_radius). A random radius is sampled each call.
    Returns:
        A new PIL Image blurred with a Gaussian kernel.
    """
    r = random.uniform(*radius_range) # Sample blur radius (higher => blurrier)
    return pil_img.filter(ImageFilter.GaussianBlur(radius=r)) # PIL handles the filtering internally


def add_jpeg(pil_img, quality_range=(10, 40)):
    """Simulate JPEG compression artifacts by re-encoding the image.
    This "damages" the image by saving it as JPEG at a low quality setting,
    then reading it back.
    Args:
        pil_img: Input PIL Image.
        quality_range: Tuple (min_quality, max_quality). Random quality is picked each call.
            Lower quality => more artifacts/blockiness.
    Returns:
        A new PIL Image after JPEG re-encode/decode.
    """
    q = random.randint(*quality_range) # Choose JPEG quality (int). Typical JPEG quality is 1-95.
    buf = io.BytesIO() # Use an in-memory buffer so we don't write any files to disk
    pil_img.save(buf, format="JPEG", quality=q) # Save as JPEG into the buffer
    buf.seek(0) # Rewind buffer and load it back as an image
    return Image.open(buf).convert("RGB") # Convert to RGB to ensure consistent 3-channel output


def add_occlusion(pil_img, frac_range=(0.08, 0.25)):
    """Occlude (cover) a random rectangular region with black pixels.
    Args:
        pil_img: Input PIL Image.
        frac_range: Tuple (min_frac, max_frac). Random fraction of total image area
            to occlude each call.
    Returns:
        A new PIL Image with a black rectangle occlusion.
    """
    img = np.array(pil_img).copy()  # Convert to NumPy so we can modify pixel values
    h, w = img.shape[:2]
    frac = random.uniform(*frac_range) # Choose how much area to occlude (as a fraction of total pixels)
    area = int(frac * h * w)
    rect_w = random.randint(int(0.1 * w), int(0.5 * w)) # Choose a random rectangle width (10% to 50% of image width)
    rect_h = max(1, area // max(1, rect_w))# Compute rectangle height so width*height ≈ desired area
    rect_h = min(rect_h, h)  # Make sure rectangle height is not bigger than the image

    # Random top-left corner so the rectangle fits inside the image
    x0 = random.randint(0, max(0, w - rect_w))
    y0 = random.randint(0, max(0, h - rect_h))

    # Set that region to black (0) for all channels
    img[y0 : y0 + rect_h, x0 : x0 + rect_w, :] = 0
    return Image.fromarray(img)


def corrupt_randomly(pil_img):
    """Apply exactly one randomly-chosen corruption to the image."""
    # Pick one corruption function at random and apply it
    return random.choice([add_noise, add_blur, add_jpeg, add_occlusion])(pil_img)