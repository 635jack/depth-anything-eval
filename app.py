import streamlit as st
import streamlit.components.v1 as components
import yaml
import os
import subprocess
import pandas as pd
import base64

# Config
CONFIG_PATH = "config.yaml"
RESULTS_DIR = "output/results"
MESHES_DIR = "output/meshes"
REPORT_PATH = os.path.join(RESULTS_DIR, "report.html")

st.set_page_config(layout="wide", page_title="Cup Generator & Evaluator")

def load_config():
    if not os.path.exists(CONFIG_PATH):
        return {"variants": []}
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, sort_keys=False)

def run_pipeline():
    with st.spinner("Running full pipeline (Blender Rendering + DA-V2 Inference)..."):
        # We can just call main.py as a subprocess
        result = subprocess.run(["./venv/bin/python", "main.py"], capture_output=True, text=True)
        if result.returncode == 0:
            st.success("Pipeline completed successfully!")
            # st.code(result.stdout)
        else:
            st.error("Pipeline execution error.")
            st.code(result.stderr)
def generate_variant_name(r_base, r_top, height, bulge, material, deformations):
    """Generates a descriptive name for the cup variant."""
    # Convert from meters back to cm for readability in the name
    rb_cm = r_base * 100
    rt_cm = r_top * 100
    h_cm = height * 100
    b_cm = bulge * 100
    
    name_parts = [f"cup_R{rb_cm:.1f}_T{rt_cm:.1f}_H{h_cm:.1f}"]
    if abs(b_cm) > 0.01:
        name_parts.append(f"B{b_cm:.1f}")
    name_parts.append(material)
    
    for d in deformations:
        name_parts.append(d['type'])
        
    return "_".join(name_parts)

# --- Sidebar: Addition Form ---
st.sidebar.header("Create New Cup")
with st.sidebar.form("new_cup_form", clear_on_submit=True):
    r_base = st.slider("Base Radius (cm)", 2.0, 8.0, 4.0) / 100.0
    r_top = st.slider("Top Radius (cm)", 2.0, 8.0, 5.0) / 100.0
    height = st.slider("Height (cm)", 3.0, 15.0, 8.0) / 100.0
    bulge = st.slider("Profile/Bulge (cm)", -3.0, 3.0, 0.0) / 100.0
    material = st.selectbox("Material (Filament)", ["matte", "silk", "translucent"])
    layer_h = st.slider("Layer Height (mm)", 0.05, 0.4, 0.2) / 10.0
    
    st.subheader("Deformations")
    add_dent = st.checkbox("Add localized dent")
    shear_x = st.slider("Shear X (Tilt)", -0.5, 0.5, 0.0)
    flare_factor = st.slider("Flare Factor (Anamorphosis)", 0.5, 2.0, 1.0)
    
    submitted = st.form_submit_button("Add to List")
    if submitted:
        config = load_config()
        deform = []
        if add_dent:
            deform.append({"type": "dent", "center_xyz": [3.0, 0, height*100/2], "radius": 2.5, "depth": 0.8})
        if abs(shear_x) > 0.001:
            deform.append({"type": "shear", "shear_x": shear_x, "shear_y": 0.0})
        if abs(flare_factor - 1.0) > 0.001:
            deform.append({"type": "flare", "factor": flare_factor})
        
        # Auto-generate name
        name = generate_variant_name(r_base, r_top, height, bulge, material, deform)
        
        # Check for duplicates and add suffix if needed
        existing_names = [v['name'] for v in config['variants']]
        base_name = name
        counter = 1
        while name in existing_names:
            name = f"{base_name}_{counter}"
            counter += 1

        config["variants"].append({
            "name": name, "r_base": r_base, "r_top": r_top, "height": height,
            "bulge": bulge, "material": material, "layer_height": layer_h,
            "deformations": deform
        })
        save_config(config)
        st.sidebar.success(f"Variant '{name}' added!")

# --- Main Layout ---
st.title("Cup Generator & Evaluation Dashboard")

config = load_config()
# st.write("### Vos Variantes Actuelles")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Variant List")
    if config["variants"]:
        df = pd.DataFrame(config["variants"])
        st.dataframe(df.drop(columns=['deformations'], errors='ignore'), height=300)
        
        to_delete = st.selectbox("Delete a variant:", ["---"] + [v['name'] for v in config['variants']])
        if st.button("Delete") and to_delete != "---":
            config["variants"] = [v for v in config["variants"] if v['name'] != to_delete]
            save_config(config)
            st.rerun()
    else:
        st.info("No variants configured. Use the sidebar to create one.")

with col2:
    st.subheader("Pipeline Control")
    if st.button("Run Full Evaluation", type="primary"):
        run_pipeline()
        st.rerun()

st.markdown("---")
st.subheader("Latest Results")

if os.path.exists(REPORT_PATH):
    # Instead of embedding raw report, let's show download links for current variants
    st.info("The models and ground-truth maps are available for download below.")
    
    # Grid of results
    if config["variants"]:
        for idx, var in enumerate(config["variants"]):
            vn = var['name']
            # Make sure vn is unique for the expander if there are duplicates in config
            expander_title = f"Variant: {vn} ({var['material']}) [#{idx}]"
            with st.expander(expander_title, expanded=True):
                c_dl, c_3d, c_viz = st.columns([1, 1, 2])
                
                with c_dl:
                    st.write("**Downloads:**")
                    stl_clean = os.path.join(MESHES_DIR, f"{vn}_clean.stl")
                    obj_print = os.path.join(MESHES_DIR, f"{vn}_printed.obj")
                    gt_exr = f"output/renders/{vn}_front.exr"
                    
                    if os.path.exists(stl_clean):
                        with open(stl_clean, "rb") as f:
                            st.download_button(f"Download Clean STL", f, file_name=f"{vn}.stl", key=f"dl_stl_{vn}_{idx}")
                    if os.path.exists(obj_print):
                        with open(obj_print, "rb") as f:
                            st.download_button(f"Download Printed OBJ", f, file_name=f"{vn}_printed.obj", key=f"dl_obj_{vn}_{idx}")
                    if os.path.exists(gt_exr):
                        with open(gt_exr, "rb") as f:
                            st.download_button(f"Download EXR Depth GT", f, file_name=f"{vn}.exr", key=f"dl_exr_{vn}_{idx}")
                
                with c_3d:
                    st.write("**3D Preview:**")
                    glb_path = os.path.join(MESHES_DIR, f"{vn}_printed.glb")
                    if os.path.exists(glb_path):
                        with open(glb_path, "rb") as f:
                            b64 = base64.b64encode(f.read()).decode()
                        
                        mv_html = f"""
                        <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.4.0/model-viewer.min.js"></script>
                        <model-viewer src="data:model/gltf-binary;base64,{b64}" 
                                      auto-rotate camera-controls 
                                      style="width: 100%; height: 250px; background-color: #f0f2f6; border-radius: 10px;">
                        </model-viewer>
                        """
                        components.html(mv_html, height=260)
                    else:
                        st.info("3D not available")

                with c_viz:
                    st.write("**DA-V2 Evaluation:**")
                    fig_path = os.path.join("output/figures", f"{vn}_front_viz.png")
                    if os.path.exists(fig_path):
                        st.image(fig_path, use_container_width=True)
                    else:
                        st.warning("Evaluation not generated.")

else:
    st.warning("No evaluation report found. Please click 'Run Full Evaluation'.")
