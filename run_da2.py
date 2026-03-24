import os
import glob
import numpy as np
import torch
from PIL import Image
from transformers import pipeline

def main():
    renders_dir = os.path.join("output", "renders")
    predictions_dir = os.path.join("output", "predictions")
    os.makedirs(predictions_dir, exist_ok=True)

    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    print("Loading Depth-Anything-V2-Large-hf pipeline...")
    pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf", device=device)

    img_paths = sorted(glob.glob(os.path.join(renders_dir, "*.png")))
    if not img_paths:
        print("No PNG images found in", renders_dir)
        return

    print(f"Found {len(img_paths)} images to process.")

    for i, path in enumerate(img_paths):
        basename = os.path.basename(path)
        name_no_ext = os.path.splitext(basename)[0]
        out_path = os.path.join(predictions_dir, f"{name_no_ext}.npy")

        try:
            image = Image.open(path).convert("RGB")
            depth_tensor = pipe(image)["predicted_depth"]
            depth_np = depth_tensor.squeeze().cpu().numpy()
            
            np.save(out_path, depth_np)
            print(f"[{i+1}/{len(img_paths)}] Saved prediction for {basename}")
        except Exception as e:
            print(f"Failed to process {basename}: {e}")

if __name__ == "__main__":
    main()
