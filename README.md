# Cup Generator & Depth-Anything-V2 Evaluator

An interactive pipeline to generate, simulate, and evaluate Depth-Anything-V2 performance on parametric 3D printed objects.

<video src="demo.mp4" autoplay loop muted playsinline width="100%"></video>

[https://github.com/635jack/cup-generator-depth-eval](https://github.com/635jack/cup-generator-depth-eval)

## Key Features

*   **Interactive Dashboard**: A Streamlit-based web interface to design custom cup variants in real-time.
*   **Parametric 3D Generation**: Automated creation of manifold cups with adjustable base radius, top radius, height, and wall thickness.
*   **Advanced Deformations**: Apply Anamorphic distortions like Shear (Z-axis tilt) and Flare (non-linear tapered growth/pinching) for complex geometric testing.
*   **3D Print Simulation**: 
    *   **Layer Stratification**: Simulates 0.2mm FDM layer lines via radial sine-wave displacement for realistic depth artifacts.
    *   **Dual-Mesh Export**: Generates both a Clean STL (for actual printing) and a Printed OBJ/GLB (for simulation).
*   **Automated Pipeline**: One-click execution of high-fidelity Blender renderings (Cycles), AI depth inference, and metric computation.

## Evaluation Workflow

1.  **Draft**: Set cup parameters and deformations in the dashboard.
2.  **Generate**: The system builds the 3D meshes and verifies manifoldness/printability.
3.  **Render**: Blender CLI generates RGB images and 32-bit EXR ground-truth depth maps (in real meters).
4.  **Inference**: Depth-Anything-V2 (Large) predicts relative depth from the RGB render.
5.  **Evaluate**: Predictions are aligned to Ground Truth to compute standard metrics: AbsRel, RMSE, δ1.25, etc.

## 🛠️ Setup

- **Python ≥ 3.10**
- **Blender ≥ 3.6** (must be in system PATH or configured in `main.py`)
- **Virtual Env**:
  ```bash
  python -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  ```

## 🏃 Usage

Launch the interactive dashboard to manage your variants and run the pipeline:
```bash
streamlit run app.py
```

## 📂 Project Structure

- `app.py`: Interactive Streamlit dashboard.
- `generate_objects.py`: Parametric mesh engine with deformation logic.
- `render_scene.py`: Blender Python script for material simulation and EXR export.
- `main.py`: Pipeline orchestrator.
- `output/`: Generated meshes (STL/OBJ), renders, and evaluation reports.

---
Created as part of ISIR-Stage research.
