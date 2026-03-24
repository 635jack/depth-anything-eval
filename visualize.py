import os
import glob
import json
import numpy as np
import pandas as pd
import OpenEXR
import Imath
import cv2
import matplotlib.pyplot as plt
from PIL import Image

def generate_error_map(gt, pred, mask):
    # Flatten
    gt_masked = gt[mask]
    pred_masked = pred[mask]

    # Align pred to 1/gt using least squares to find scale and shift.
    inv_gt = 1.0 / gt_masked
    A = np.vstack([pred_masked, np.ones_like(pred_masked)]).T
    
    try:
        s, t = np.linalg.lstsq(A, inv_gt, rcond=None)[0]
    except:
        s, t = 1.0, 0.0

    aligned_inv_depth = s * pred + t
    aligned_inv_depth = np.clip(aligned_inv_depth, a_min=1e-5, a_max=None)
    pred_aligned = 1.0 / aligned_inv_depth
    
    error = np.abs(gt - pred_aligned)
    
    # Hide error outside mask
    error[~mask] = 0
    pred_aligned[~mask] = 0
    gt_viz = gt.copy()
    gt_viz[~mask] = 0
    
    return gt_viz, pred_aligned, error, s, t

def main():
    renders_dir = os.path.join("output", "renders")
    predictions_dir = os.path.join("output", "predictions")
    results_dir = os.path.join("output", "results")
    figures_dir = os.path.join("output", "figures")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    prediction_files = sorted(glob.glob(os.path.join(predictions_dir, "*.npy")))
    
    html_content = [
        "<html><head><title>Depth-Anything-V2 Evaluation</title>",
        "<style>body {font-family: Arial, sans-serif; margin: 20px;} ",
        "table {border-collapse: collapse; width: 100%;} th, td {border: 1px solid #ddd; padding: 8px;} th {background-color: #f2f2f2;} ",
        "img {max-width: 100%; height: auto;}</style></head><body>",
        "<h1>Depth-Anything-V2 Evaluation Report</h1>"
    ]
    
    csv_path = os.path.join(results_dir, "metrics_detailed.csv")
    metrics_html = ""
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        metrics_html = "<h2>Metrics (Detailed)</h2>" + df.to_html(index=False)
        html_content.append(metrics_html)
    
    html_content.append("<h2>Visualizations</h2>")
    html_content.append("<table><tr><th>RGB</th><th>GT Depth</th><th>Pred (Aligned)</th><th>Absolute Error (hot)</th></tr>")

    for pred_path in prediction_files:
        basename = os.path.basename(pred_path)
        name_no_ext = os.path.splitext(basename)[0]
        
        rgb_path = os.path.join(renders_dir, f"{name_no_ext}.png")
        gt_path = os.path.join(renders_dir, f"{name_no_ext}.exr")
        
        if not os.path.exists(gt_path) or not os.path.exists(rgb_path):
            continue
            
        # Use OpenEXR to read EXR 32bit float
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        try:
            exr = OpenEXR.InputFile(gt_path)
            dw = exr.header()['dataWindow']
            size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
            channels = exr.header()['channels'].keys()
            
            target_ch = list(channels)[0]
            for ch in ['V', 'Z', 'Y', 'Depth', 'R']:
                if ch in channels:
                    target_ch = ch
                    break
            
            z_str = exr.channel(target_ch, pt)
            gt_depth = np.frombuffer(z_str, dtype=np.float32).copy()
            gt_depth.shape = (size[1], size[0])
        except Exception as e:
            print(f"Failed to load EXR {gt_path}: {e}")
            continue
            
        pred_raw = np.load(pred_path)
        rgb = np.array(Image.open(rgb_path).convert("RGB"))
        
        if pred_raw.shape != gt_depth.shape:
            pred_raw = cv2.resize(pred_raw, (gt_depth.shape[1], gt_depth.shape[0]), interpolation=cv2.INTER_LINEAR)
            
        mask = (gt_depth > 0) & (gt_depth < 2.0)
        
        if mask.sum() < 100:
            continue
            
        gt_viz, pred_viz, error, s, t = generate_error_map(gt_depth, pred_raw, mask)
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(rgb)
        axes[0].set_title("RGB")
        axes[0].axis('off')
        
        vmax_depth = np.percentile(gt_viz[mask], 99) if mask.sum() > 0 else 2.0
        
        im1 = axes[1].imshow(gt_viz, cmap='viridis', vmin=0, vmax=vmax_depth)
        axes[1].set_title("Ground Truth (m)")
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        im2 = axes[2].imshow(pred_viz, cmap='viridis', vmin=0, vmax=vmax_depth)
        axes[2].set_title(f"Pred Aligned\ns={s:.2e}, t={t:.2e}")
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
        vmax_err = np.percentile(error[mask], 95) if mask.sum() > 0 else 0.5
        im3 = axes[3].imshow(error, cmap='hot', vmin=0, vmax=vmax_err)
        axes[3].set_title("Absolute Error (m)")
        axes[3].axis('off')
        plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        out_fig_path = os.path.join(figures_dir, f"{name_no_ext}_viz.png")
        plt.savefig(out_fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        html_content.append(f"""
        <tr>
            <td colspan="4" style="text-align:center;font-weight:bold;background-color:#eee;">{name_no_ext}</td>
        </tr>
        <tr>
            <td colspan="4"><img src="../figures/{name_no_ext}_viz.png" width="100%"></td>
        </tr>
        """)
        
    html_content.append("</table></body></html>")
    
    report_path = os.path.join(results_dir, "report.html")
    with open(report_path, "w") as f:
        f.write("\n".join(html_content))
        
    print(f"Generated report at {report_path}")

if __name__ == "__main__":
    main()
