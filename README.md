# Image-Corruption-Detector
(AI/ML) PyTorch computer-vision mini project that detects clean vs corrupted images. Generates synthetic corruptions (noise/blur/JPEG/occlusion) on CIFAR-10, trains a CNN, evaluates accuracy/F1, and includes a CLI to score real images with confidence.

# Image Corruption Detector (Clean vs Corrupted)
A PyTorch computer-vision mini project that detects whether an image is **clean** or **corrupted**.  
Corruptions are generated **on-the-fly** (noise, blur, JPEG compression artifacts, and occlusion). The project includes:

- ✅ Training pipeline (CIFAR-10)
- ✅ Evaluation (Accuracy, F1, confusion matrix)
- ✅ Inference CLI to score your own images with confidence
- ✅ Corruption preview script that saves before/after examples

---


## Why this project matters
Real-world vision systems (camera pipelines, CV models, OCR, etc.) can degrade when images are:
- blurred (motion/out-of-focus)
- noisy (low light / sensor noise)
- compressed (JPEG artifacts)
- partially missing/blocked (occlusion)

Detecting corrupted inputs is useful for:
- rejecting low-quality images
- triggering re-capture / preprocessing
- routing to fallback models or safer paths
---

## Results
> Note: corruptions are generated randomly each run, so results can vary slightly.
Example run:
- **Test Accuracy:** ~98%
- **Test F1:** ~98%
- Confusion matrix example:
  - `[[4971, 98], [89, 4842]]`



## Tech Stack

- **Python 3.12**
- **PyTorch** + **torchvision**
- **NumPy**
- **Pillow (PIL)**
- **scikit-learn** (accuracy, F1, confusion matrix)
- **tqdm** (progress bars)
- Runs on Apple Silicon GPU via **MPS** when available

## Repo Structure
src/
corruptions.py           # noise/blur/jpeg/occlusion functions
data.py                  # CIFAR-10 dataset wrapper (clean vs corrupted labels)
train.py                 # training loop + saves model weights
eval.py                  # evaluation: accuracy, F1, confusion matrix
infer.py                 # CLI inference on user images with confidence
preview_corruptions.py   # saves clean + corrupted previews to outputs/
data/                      # CIFAR-10 downloaded files (optional to keep in repo)
models/                    # trained model weights (.pt)
examples/                  # your test images for inference
outputs/                   # preview images created by preview_corruptions.py



---

## Setup

### Option A: Conda 
```bash
    conda create -n amdml python=3.12 -y
    conda activate amdml
    pip install -r requirements.txt

If you don’t have requirements.txt, install directly:
    pip install torch torchvision numpy pillow tqdm scikit-learn

Train

Trains a small CNN to classify:
	•	0 = clean
	•	1 = corrupted

    python -m src.train

Outputs:
	•	downloads CIFAR-10 (if not already)
	•	prints epoch loss + validation accuracy
	•	saves model to:
	•	models/clean_vs_corrupt_cnn.pt


Evaluate
Runs on CIFAR-10 test split and prints:
	•	Accuracy
	•	F1 score
	•	Confusion matrix

    python -m src.eval

Inference on Your Own Images (CLI)
	1.	Put images into examples/ (jpg/png)
	2.	Run: 
        bash: 
            python -m src.infer --input_dir examples

Example output:
Device: mps
photo1.jpg  ->  corrupted  (confidence=0.818)
photo2.jpg  ->  clean      (confidence=0.970)

!!!!
The model was trained on CIFAR-10 (32×32 images). Real photos are resized to 32×32, so some results may be imperfect—this is expected for a mini-project and still demonstrates the full ML pipeline.
!!!!


Visualize Corruptions (Preview Script)

This creates and saves:
	•	clean image
	•	noise
	•	blur
	•	jpeg artifacts
	•	occlusion

    bash: 
        python -m src.preview_corruptions

Outputs saved to:
	•	outputs/previews/


How the Dataset Works (On-the-fly Corruption)

Instead of permanently saving corrupted copies, the dataset wrapper applies corruptions in __getitem__:
	•	with probability p_corrupt (default 0.5), apply 1 random corruption → label 1
	•	otherwise keep image clean → label 0

This means the model sees slightly different corruptions each epoch, which helps robustness.



Notes on Large Files (Git LFS)
This repo may include large assets (dataset tarball, model weights). If cloning/pulling, ensure Git LFS is installed:
bash: 
    git lfs install
    git lfs pull

    
