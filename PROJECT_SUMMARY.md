# 🥤 Cup Generator & Evaluation Dashboard — Project Summary

This project is a complete pipeline for **synthetic data generation** and **Depth Estimation Evaluation** using **Depth-Anything-V2 (DA-V2)**. It specifically focuses on evaluating depth models against realistic 3D-printed objects (cups).

## 🚀 Core Objective
To simulate a real-world physical fabrication workflow and evaluate how depth estimation models perform on objects with complex geometries and 3D printing artifacts (layer lines, varied materials).

---

## 🛠️ The Construction: Step-by-Step

### 1. Parametric 3D Generation (`generate_objects.py`)
- **Base Geometry**: Generates ellipsoidal cups based on Base/Top Radius, Height, and Bulge.
- **Deformations**:
    - **Local Dent**: Simulates impacts or physical defects.
    - **Anamorphic Shear**: Tilts the object along Z for perspective distortion.
    - **Anamorphic Flare**: Non-linear expansion towards the top.
- **Manifold Verification**: Ensures all meshes are watertight and printable (manifold, minimal thickness, flat base).

### 2. Physical Printing Simulation
- **Layer Stratification**: Applies a radial sine-wave displacement (0.2mm layers) to simulate the texture of FDM 3D printing.
- **Dual-Mesh Export**: Every variant creates a `clean.stl` (pure geo for printing) and a `printed.obj/glb` (textured geo for simulation).

### 3. Rendering Pipeline (`render_scene.py` via Blender)
- **Materials Factory**: Simulates **Matte PLA**, **Silk PLA** (anisotropic), and **Translucent/PETG** (refractive) using Blender's Cycles.
- **Ground Truth**: High-precision 32-bit Float EXR depth maps.

### 4. Inference & Evaluation
- **DA-V2 Inference**: Runs `run_da2.py` on rendered images to produce relative depth.
- **Metric Computation (`metrics.py`)**: Aligns predicted depth to ground truth (least-squares scale/shift) and calculates:
    - **AbsRel, RMSE, SqRel, RMSE_log**
    - **Threshold Metrics (d1, d2, d3)**

### 5. Interactive Dashboard (`app.py`)
- **Variant Creator**: UI to adjust cup parameters and deformations.
- **Real-time Preview**: Fast Numpy-based depth rendering + 3D interactive viewer (`model-viewer`).
- **Printability Guard**: Automatic warning if overhang angles > 45° (supports required).
- **Download Center**: Direct access to STL, OBJ, and EXR files for physical printing and analysis.

---

## 📂 Key File Structure
- `app.py`: Streamlit Dashboard (Main UI).
- `main.py`: Orchestrator (Sequences the whole pipeline).
- `generate_objects.py`: 3D Meshing and verification logic.
- `render_scene.py`: Blender rendering automation.
- `metrics.py`: Alignment and performance scoring.
- `visualize.py`: Static HTML report generator.
- `config.yaml`: The source of truth for all cup variants.

## 🏃 How to Start
1. Ensure the virtual environment is ready: `source venv/bin/activate`
2. Run the dashboard: `./venv/bin/streamlit run app.py`
3. Access at: [http://localhost:8501](http://localhost:8501)
