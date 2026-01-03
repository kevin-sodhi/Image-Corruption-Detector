import os
import argparse
import torch
from torchvision import transforms
from PIL import Image

from .train import SmallCNN, get_device

LABELS = ["clean", "corrupted"]

def load_model(model_path: str, device: str):
    model = SmallCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/clean_vs_corrupt_cnn.pt", help="Path to .pt model")
    parser.add_argument("--input_dir", default="examples", help="Folder of images to score")
    args = parser.parse_args()

    device = get_device()
    print("Device:", device)

    preprocess = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
    ])

    model = load_model(args.model, device)

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"Input dir not found: {args.input_dir}")

    files = [f for f in os.listdir(args.input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not files:
        print(f"No images found in {args.input_dir}. Add .png/.jpg files and run again.")
        return

    for fname in sorted(files):
        path = os.path.join(args.input_dir, fname)
        img = Image.open(path).convert("RGB")
        x = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred = int(probs.argmax())
            conf = float(probs[pred])

        print(f"{fname:30s}  ->  {LABELS[pred]:9s}  (confidence={conf:.3f})")

if __name__ == "__main__":
    main()
