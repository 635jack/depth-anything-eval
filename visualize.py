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
        '<script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.4.0/model-viewer.min.js"></script>',
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
    html_content.append("<table><tr><th style='width: 350px;'>Modèle 3D Interactif</th><th>Évaluation Pipeline (RGB | GT | Pred | Erreur)</th></tr>")

    # Pass 1: Compute global scales and cache data
    global_vmax_depth = 0.0
    global_vmax_err = 0.0
    cache = []

    for pred_path in prediction_files:
        basename = os.path.basename(pred_path)
        name_no_ext = os.path.splitext(basename)[0]
        
        rgb_path = os.path.join(renders_dir, f"{name_no_ext}.png")
        gt_path = os.path.join(renders_dir, f"{name_no_ext}.exr")
        unc_path = pred_path.replace(".npy", "_unc.npy")
        
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
        
        # Update globals
        global_vmax_depth = max(global_vmax_depth, np.percentile(gt_viz[mask], 99))
        global_vmax_err = max(global_vmax_err, np.percentile(error[mask], 99))
            
        cache.append({
            'name': name_no_ext, 'rgb': rgb, 'gt': gt_viz, 'pred': pred_viz,
            'err': error, 's': s, 't': t
        })

    # Group cache by object name
    objects_dict = {}
    for item in cache:
        obj_name = item['name'].rsplit("_", 1)[0]
        if obj_name not in objects_dict:
            objects_dict[obj_name] = []
        objects_dict[obj_name].append(item)

    import base64

    # Pass 2: Plotting with global scales
    for obj_name, items in objects_dict.items():
        rowspan = len(items) * 2
        
        # Load GLB as base64 to bypass local file CORS policy in browsers
        glb_path = os.path.join("output", "meshes", f"{obj_name}.glb")
        if os.path.exists(glb_path):
            with open(glb_path, "rb") as f:
                b64_data = base64.b64encode(f.read()).decode('utf-8')
            glb_src = f"data:model/gltf-binary;base64,{b64_data}"
        else:
            glb_src = f"../meshes/{obj_name}.glb"
        
        html_content.append("<tr>")
        html_content.append(f'<td rowspan="{rowspan}" style="text-align:center; vertical-align:middle; border-right:2px solid #ddd;">')
        html_content.append(f"<h3>{obj_name}</h3>")
        html_content.append(f'<model-viewer src="{glb_src}" auto-rotate camera-controls style="width: 100%; height: 350px; background-color: #f9f9f9; border-radius: 8px;"></model-viewer>')
        html_content.append('<p style="font-size: 0.85em; color: #666; margin-top: 8px;">(Faites glisser pour tourner)</p>')
        html_content.append('</td>')

        for i, item in enumerate(items):
            if i > 0:
                html_content.append("<tr>")
            
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
            axes[0].imshow(item['rgb'])
            axes[0].set_title("RGB")
            axes[0].axis('off')
            
            im1 = axes[1].imshow(item['gt'], cmap='viridis', vmin=0, vmax=global_vmax_depth)
            axes[1].set_title("Ground Truth (m)")
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            
            im2 = axes[2].imshow(item['pred'], cmap='viridis', vmin=0, vmax=global_vmax_depth)
            axes[2].set_title(f"Pred Aligned\ns={item['s']:.2e}, t={item['t']:.2e}")
            axes[2].axis('off')
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
            
            im3 = axes[3].imshow(item['err'], cmap='hot', vmin=0, vmax=global_vmax_err)
            axes[3].set_title("Absolute Error (m)")
            axes[3].axis('off')
            plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            
            out_fig_path = os.path.join(figures_dir, f"{item['name']}_viz.png")
            plt.savefig(out_fig_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            html_content.append(f"""
                <td style="text-align:center;font-weight:bold;background-color:#eee;">{item['name']}</td>
            </tr>
            <tr>
                <td><img src="../figures/{item['name']}_viz.png" width="100%"></td>
            </tr>
            """)
        
    html_content.append("</table></body></html>")
    
    report_path = os.path.join(results_dir, "report.html")
    with open(report_path, "w") as f:
        f.write("\n".join(html_content))
        
    print(f"Generated report at {report_path}")

if __name__ == "__main__":
    main()
