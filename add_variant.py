#!/usr/bin/env python3
import yaml
import os

def main():
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        config = {"variants": []}
    else:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

    print("--- Adding a New Cup Variant ---")
    name = input("Variant name (e.g., cup_custom_1): ")
    r_base = float(input("Base radius (m, e.g., 0.04): "))
    r_top = float(input("Top radius (m, e.g., 0.05): "))
    height = float(input("Height (m, e.g., 0.10): "))
    bulge = float(input("Bulge/Profile (m, e.g., 0.01 for barrel, -0.01 for pinched): "))
    
    print("\nFilament choice:")
    print("1. Matte")
    print("2. Silk (Glossy)")
    print("3. Translucent (PETG)")
    mat_choice = input("Your choice (1/2/3): ")
    mat_map = {"1": "matte", "2": "silk", "3": "translucent"}
    material = mat_map.get(mat_choice, "matte")
    
    layer_h = float(input("Layer height (cm, e.g., 0.02): "))

    new_variant = {
        "name": name,
        "r_base": r_base,
        "r_top": r_top,
        "height": height,
        "bulge": bulge,
        "material": material,
        "layer_height": layer_h,
        "deformations": []
    }

    # Optional deformation
    add_dent = input("Add a dent (y/n)? ")
    if add_dent.lower() == 'y':
        new_variant["deformations"].append({
            "type": "dent",
            "center_xyz": [3.0, 0, height*100/2],
            "radius": 2.5,
            "depth": 0.8
        })

    config["variants"].append(new_variant)

    with open(config_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    print(f"\n✅ Variant '{name}' added to config.yaml.")
    print("⚠️  Run './venv/bin/python main.py' to update the report.")

if __name__ == "__main__":
    main()
