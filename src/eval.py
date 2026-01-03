import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from .data import CleanVsCorruptDataset
from .train import SmallCNN, get_device

def main(model_path="models/clean_vs_corrupt_cnn.pt", batch_size=256):
    device = get_device()
    print("Device:", device)

    ds = CleanVsCorruptDataset(train=False, p_corrupt=0.5)  # test split
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = SmallCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu()
            y_true.extend(y.tolist())
            y_pred.extend(pred.tolist())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test F1:       {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    main()
