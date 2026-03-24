import os
import glob
import numpy as np
import pandas as pd
import OpenEXR
import Imath
import cv2

def compute_metrics(gt, pred, mask):
    # Flatten
    gt = gt[mask]
    pred = pred[mask]

    # DA-V2 predicts inverse depth (disparity). 
    # Align pred to 1/gt using least squares to find scale and shift.
    # 1/gt = s * pred + t
    inv_gt = 1.0 / gt
    A = np.vstack([pred, np.ones_like(pred)]).T
    s, t = np.linalg.lstsq(A, inv_gt, rcond=None)[0]
    
    # Get aligned inverse depth and convert to depth
    aligned_inv_depth = s * pred + t
    # Prevent divide by zero or negative depth
    aligned_inv_depth = np.clip(aligned_inv_depth, a_min=1e-5, a_max=None)
    pred_aligned = 1.0 / aligned_inv_depth

    # Compute metrics
    thresh = np.maximum((gt / pred_aligned), (pred_aligned / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred_aligned) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred_aligned)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred_aligned) / gt)
    sq_rel = np.mean(((gt - pred_aligned) ** 2) / gt)

    return {
        "AbsRel": abs_rel,
        "SqRel": sq_rel,
        "RMSE": rmse,
        "RMSElog": rmse_log,
        "d1": d1,
        "d2": d2,
        "d3": d3
    }, s, t

def main():
    renders_dir = os.path.join("output", "renders")
    predictions_dir = os.path.join("output", "predictions")
    results_dir = os.path.join("output", "results")
    os.makedirs(results_dir, exist_ok=True)

    prediction_files = glob.glob(os.path.join(predictions_dir, "*.npy"))
    
    results = []

    for pred_path in prediction_files:
        basename = os.path.basename(pred_path)
        name_no_ext = os.path.splitext(basename)[0]
        
        # Ground truth EXR path
        gt_path = os.path.join(renders_dir, f"{name_no_ext}.exr")
        if not os.path.exists(gt_path):
            print(f"Skipping {name_no_ext}, GT EXR not found.")
            continue
            
        # Load GT
        # Use OpenEXR to read EXR 32bit float
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        try:
            exr = OpenEXR.InputFile(gt_path)
            dw = exr.header()['dataWindow']
            size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
            channels = exr.header()['channels'].keys()
            
            # Find the best channel
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
            
        # Load Pred
        pred_raw = np.load(pred_path)
        
        # Resize pred to match GT if needed (DA-V2 usually handles standard sizes but just in case)
        if pred_raw.shape != gt_depth.shape:
            pred_raw = cv2.resize(pred_raw, (gt_depth.shape[1], gt_depth.shape[0]), interpolation=cv2.INTER_LINEAR)
            
        # Valid mask: gt > 0 and gt < 2m. Blender background is often 100 or very large if infinite.
        mask = (gt_depth > 0) & (gt_depth < 2.0)
        
        if mask.sum() < 100:
            print(f"[{name_no_ext}] Too few valid pixels ({mask.sum()}). Skipping.")
            continue
            
        metrics, s, t = compute_metrics(gt_depth, pred_raw, mask)
        
        # Parse info from name: e.g. "box_simple_side45"
        parts = name_no_ext.rsplit("_", 1)
        obj_name = parts[0]
        angle = parts[1] if len(parts) > 1 else "unknown"
        
        res = {
            "Object": obj_name,
            "Angle": angle,
            "Scale": s,
            "Shift": t
        }
        res.update(metrics)
        results.append(res)
        
    if not results:
        print("No valid results computed.")
        return
        
    df = pd.DataFrame(results)
    
    # Save detailed CSV
    csv_path = os.path.join(results_dir, "metrics_detailed.csv")
    df.to_csv(csv_path, index=False)
    
    # Compute global averages
    print("\n--- GLOBAL METRICS ---")
    numeric_cols = ["AbsRel", "SqRel", "RMSE", "RMSElog", "d1", "d2", "d3"]
    mean_metrics = df[numeric_cols].mean()
    for k, v in mean_metrics.items():
        print(f"{k}: {v:.4f}")
        
    df_agg = df.groupby("Angle")[numeric_cols].mean()
    print("\n--- METRICS BY ANGLE ---")
    print(df_agg)

    df_agg_obj = df.groupby("Object")[numeric_cols].mean()
    print("\n--- METRICS BY OBJECT ---")
    print(df_agg_obj)

    print(f"\nSaved metrics to {csv_path}")

if __name__ == "__main__":
    main()
