import os
from torchvision import datasets
from .corruptions import add_noise, add_blur, add_jpeg, add_occlusion

def main(out_dir="outputs/previews"):
    os.makedirs(out_dir, exist_ok=True)

    ds = datasets.CIFAR10(root="./data", train=False, download=True)
    pil_img, _ = ds[0]

    pil_img.save(os.path.join(out_dir, "0_clean.png"))
    add_noise(pil_img).save(os.path.join(out_dir, "1_noise.png"))
    add_blur(pil_img).save(os.path.join(out_dir, "2_blur.png"))
    add_jpeg(pil_img).save(os.path.join(out_dir, "3_jpeg.png"))
    add_occlusion(pil_img).save(os.path.join(out_dir, "4_occlusion.png"))

    print(f"Saved previews to: {out_dir}")
    print("Open the folder to see clean vs corrupted images.")

if __name__ == "__main__":
    main()
