import os
import subprocess
import sys

def run_step(cmd, desc):
    print(f"\n{'='*50}\n{desc}\n{'='*50}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Error running: {desc}. Exiting.")
        sys.exit(result.returncode)

def main():
    print("Starting Depth-Anything-V2 Evaluation Pipeline")
    
    run_step(f"{sys.executable} generate_objects.py", "Generating 3D Objects")
    
    blender_exe = "/Applications/Blender.app/Contents/MacOS/Blender"
    if not os.path.exists(blender_exe):
        blender_exe = "blender" # fallback to PATH
        
    cmd_render = f'"{blender_exe}" --background --python render_scene.py -- --obj_dir output/meshes --output_dir output/renders'
    run_step(cmd_render, "Rendering Scenes in Blender")
    
    run_step(f"PYTORCH_ENABLE_MPS_FALLBACK=1 {sys.executable} run_da2.py", "Running DA-V2 Inference")
    
    run_step(f"{sys.executable} metrics.py", "Computing Metrics")
    
    run_step(f"{sys.executable} visualize.py", "Generating Visualizations and HTML Report")
    
    print("\nPipeline completed successfully!")
    print("Check output/results/report.html for the final report.")

if __name__ == "__main__":
    main()
