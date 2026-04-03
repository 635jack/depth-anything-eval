#!/usr/bin/env python3
"""
generate_objects.py — Generation of parametric 3D objects for DA-V2 evaluation.

Each object is verified (manifold, thickness, overhangs, base plate)
then exported as STL (print) and OBJ (Blender render).
"""

import os
import yaml
import numpy as np
import trimesh
from trimesh.creation import revolve
from shapely.geometry import Polygon


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "meshes")
MIN_THICKNESS_MM = 1.0        # minimum acceptable thickness (mm)
MAX_OVERHANG_ANGLE_DEG = 45   # max angle without supports


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _check_manifold(mesh: trimesh.Trimesh, name: str) -> bool:
    ok = mesh.is_watertight
    status = "✅ manifold" if ok else "⚠️  NON manifold"
    print(f"  [{name}] {status}")
    return ok


def _check_thickness(mesh: trimesh.Trimesh, name: str, min_mm: float = MIN_THICKNESS_MM) -> bool:
    """Verifies that the smallest bounding-box dimension >= min_mm."""
    extents = mesh.bounding_box.extents  # in meters
    # trimesh works in cm, convert to mm for check
    min_extent_mm = min(extents) * 10  # cm → mm
    ok = min_extent_mm >= min_mm
    status = f"✅ min thickness {min_extent_mm:.1f}mm" if ok else f"⚠️  thickness {min_extent_mm:.1f}mm < {min_mm}mm"
    print(f"  [{name}] {status}")
    return ok


def check_overhangs(mesh: trimesh.Trimesh, name: str = "obj", max_angle_deg: float = MAX_OVERHANG_ANGLE_DEG) -> dict:
    """Detects faces with an overhang > max_angle_deg relative to vertical."""
    normals = mesh.face_normals
    # Angle between face normal and "up" vector (Z+)
    up = np.array([0, 0, 1])
    cos_angles = np.dot(normals, up)
    angles_deg = np.degrees(np.arccos(np.clip(cos_angles, -1, 1)))
    # Faces pointing down (angle > 90°) with angle > (180 - max_angle)
    overhang_mask = angles_deg > (180 - max_angle_deg)
    n_overhang = int(overhang_mask.sum())
    pct = 100 * n_overhang / len(normals) if len(normals) > 0 else 0
    ok = pct < 5  # < 5% overhang faces = OK
    status = f"✅ overhangs {pct:.1f}%" if ok else f"⚠️  overhangs {pct:.1f}%"
    # Only print if called from main
    if name != "preview":
        print(f"  [{name}] {status} ({n_overhang}/{len(normals)} faces)")
    return {"ok": ok, "pct": pct, "n_faces": n_overhang, "max_angle": 180 - angles_deg.min() if len(angles_deg) > 0 else 0}


def _check_flat_base(mesh: trimesh.Trimesh, name: str) -> bool:
    """Verifies that there are faces at Z ~= z_min (flat base)."""
    z_min = mesh.bounds[0, 2]
    z_tol = 0.01  # 0.01 cm tolerance
    face_centroids_z = mesh.triangles_center[:, 2]
    n_base = int(np.sum(np.abs(face_centroids_z - z_min) < z_tol))
    ok = n_base > 0
    status = f"✅ flat base ({n_base} faces)" if ok else "⚠️  no flat base"
    print(f"  [{name}] {status}")
    return ok


def _export(mesh: trimesh.Trimesh, name: str, suffix: str, out_dir: str = OUTPUT_DIR):
    """Exports as STL, OBJ, and GLB with a suffix."""
    full_name = f"{name}_{suffix}"
    stl_path = os.path.join(out_dir, f"{full_name}.stl")
    obj_path = os.path.join(out_dir, f"{full_name}.obj")
    glb_path = os.path.join(out_dir, f"{full_name}.glb")
    mesh.export(stl_path)
    mesh.export(obj_path)
    mesh.export(glb_path)
    print(f"  [{full_name}] Extents: {mesh.bounding_box.extents} cm")
    return stl_path, obj_path, glb_path


def validate_mesh(mesh: trimesh.Trimesh, name: str) -> dict:
    """Validates the physical properties of the mesh."""
    manifold = _check_manifold(mesh, name)
    thickness = _check_thickness(mesh, name)
    overhangs = check_overhangs(mesh, name)
    flat_base = _check_flat_base(mesh, name)
    
    return {
        "manifold": manifold,
        "thickness_ok": thickness,
        "overhangs_pct": overhangs["pct"],
        "overhangs_ok": overhangs["ok"],
        "flat_base": flat_base,
    }


# ---------------------------------------------------------------------------
# Objet generators
# ---------------------------------------------------------------------------

def generate_ellipsoidal_cup(r_base, r_top, height, bulge=0.0, wall=0.003, segments=64, profile_res=200):
    """
    Generates a cup with an ellipsoidal profile by revolution.
    """
    t = np.linspace(0, 1, profile_res)
    
    # Outer profile
    r_mid = (r_base + r_top) / 2 + bulge
    r_profile = (1 - t)**2 * r_base + 2*(1-t)*t * r_mid + t**2 * r_top
    z_profile = t * height
    
    # Inner profile (stops at z=wall to have a bottom)
    t_inner = np.linspace(wall/height, 1, profile_res)
    r_inner_profile = (1 - t_inner)**2 * r_base + 2*(1-t_inner)*t_inner * r_mid + t_inner**2 * r_top
    r_inner = r_inner_profile - wall
    z_inner = t_inner * height
    
    vertices_outer = np.column_stack([r_profile, z_profile])
    vertices_inner = np.column_stack([r_inner[::-1], z_inner[::-1]])
    
    bottom_center = [[0, 0]]
    inner_center = [[0, wall]]
    
    profile = np.vstack([
        bottom_center, 
        vertices_outer, 
        vertices_inner, 
        inner_center,
        bottom_center
    ])
    
    mesh = trimesh.creation.revolve(profile, sections=segments)
    
    # Convertir en cm pour avoir des bounds logiques avec le script existant (ex: max_dim ~ 8cm = 8.0)
    mesh.apply_scale(100.0) 
    return mesh


def apply_anisotropic_scale(mesh, sx, sy, sz):
    """Applies non-uniform scale."""
    matrix = np.eye(4)
    matrix[0, 0] = sx
    matrix[1, 1] = sy
    matrix[2, 2] = sz
    mesh.apply_transform(matrix)
    return mesh


def apply_wavy_deformation(mesh, amplitude=0.2, frequency=8.0):
    """Adds waves to the edges (based on polar angle)."""
    vertices = mesh.vertices.copy()
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    
    # Polar angle calculation
    angles = np.arctan2(y, x)
    radii = np.sqrt(x**2 + y**2)
    
    # Only deform if not too close to the central axis (keep bottom intact)
    mask = radii > 0.5
    
    # Radial sinusoidal displacement
    deformation = amplitude * np.sin(frequency * angles)
    
    new_radii = radii + deformation * mask
    vertices[mask, 0] = new_radii[mask] * np.cos(angles[mask])
    vertices[mask, 1] = new_radii[mask] * np.sin(angles[mask])
    
    mesh.vertices = vertices
    return mesh


def apply_dent(mesh, center_xyz, radius=2.0, depth=0.5):
    """Creates a localized dent."""
    vertices = mesh.vertices.copy()
    dists = np.linalg.norm(vertices - center_xyz, axis=1)
    
    # Influence zone mask (inverted gaussian bell)
    mask = dists < radius
    influence = (1.0 - (dists[mask] / radius))**2
    
    # Direction towards object center (in XY)
    center_direction = -vertices[mask, :2]
    center_direction /= (np.linalg.norm(center_direction, axis=1)[:, None] + 1e-6)
    
    vertices[mask, :2] += center_direction * influence[:, None] * depth
    
    mesh.vertices = vertices
    return mesh


def apply_twist(mesh, total_angle_deg=30.0):
    """Applies a twist based on height Z."""
    vertices = mesh.vertices.copy()
    z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
    z_rel = (vertices[:, 2] - z_min) / (z_max - z_min + 1e-6)
    
    angles = np.radians(total_angle_deg) * z_rel
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    
    new_x = vertices[:, 0] * cos_a - vertices[:, 1] * sin_a
    new_y = vertices[:, 0] * sin_a + vertices[:, 1] * cos_a
    
    vertices[:, 0] = new_x
    vertices[:, 1] = new_y
    
    mesh.vertices = vertices
    return mesh


def apply_shear(mesh, shear_x=0.2, shear_y=0.0):
    """Tilts the object along the Z axis (Anamorphosis)."""
    vertices = mesh.vertices.copy()
    z_min = vertices[:, 2].min()
    z_rel = vertices[:, 2] - z_min
    
    vertices[:, 0] += z_rel * shear_x
    vertices[:, 1] += z_rel * shear_y
    
    mesh.vertices = vertices
    return mesh


def apply_flare(mesh, flare_factor=1.5):
    """Expands the radius non-linearly towards the top (Anamorphosis)."""
    vertices = mesh.vertices.copy()
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    
    z_min, z_max = z.min(), z.max()
    z_rel = (z - z_min) / (z_max - z_min + 1e-6)
    
    # Scale radius by z_rel^flare_factor
    # flare_factor > 1 -> more opening at the top
    # flare_factor < 1 -> pinched towards top
    radii = np.sqrt(x**2 + y**2)
    angles = np.arctan2(y, x)
    
    new_radii = radii * (1.0 + z_rel * (flare_factor - 1.0))
    
    vertices[:, 0] = new_radii * np.cos(angles)
    vertices[:, 1] = new_radii * np.sin(angles)
    
    mesh.vertices = vertices
    return mesh


def apply_layer_stratification(mesh, layer_height_cm=0.02, amplitude_cm=0.005):
    """Simulates FDM print layers by sinusoidal radial displacement."""
    vertices = mesh.vertices.copy()
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    
    # Angle and radius calculation
    radii = np.sqrt(x**2 + y**2)
    angles = np.arctan2(y, x)
    
    # Spatial frequency (cycles per cm)
    frequency = 1.0 / layer_height_cm
    
    # Z-based deformation (sinewave simulating the layer bulge)
    # Avoid object bottom (z < epsilon)
    mask = (z > 0.05) & (radii > 0.1)
    
    deformation = amplitude_cm * np.sin(2 * np.pi * z * frequency)
    
    new_radii = radii + (deformation * mask)
    vertices[mask, 0] = new_radii[mask] * np.cos(angles[mask])
    vertices[mask, 1] = new_radii[mask] * np.sin(angles[mask])
    
    mesh.vertices = vertices
    return mesh


# ---------------------------------------------------------------------------
# Main Logic
# ---------------------------------------------------------------------------

def generate_all() -> list[dict]:
    """Reads config.yaml and generates Clean/Printed pairs."""
    _ensure_output_dir()
    
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        return []

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    reports = []
    for var in config['variants']:
        name = var['name']
        print(f"\nProcessing Variant: {name}")
        
        # 1. Base Mesh (Clean)
        mesh = generate_ellipsoidal_cup(
            var['r_base'], var['r_top'], var['height'], 
            bulge=var.get('bulge', 0.0), 
            profile_res=400
        )
        
        # Apply deformations (if any)
        for defom in var.get('deformations', []):
            if defom['type'] == 'dent':
                mesh = apply_dent(mesh, defom['center_xyz'], defom['radius'], defom['depth'])
            elif defom['type'] == 'twist':
                mesh = apply_twist(mesh, defom['total_angle_deg'])
            elif defom['type'] == 'scale':
                mesh = apply_anisotropic_scale(mesh, defom['sx'], defom['sy'], defom['sz'])
            elif defom['type'] == 'shear':
                mesh = apply_shear(mesh, defom.get('shear_x', 0.0), defom.get('shear_y', 0.0))
            elif defom['type'] == 'flare':
                mesh = apply_flare(mesh, defom.get('factor', 1.0))

        # Validate Clean
        val = validate_mesh(mesh, f"{name}_clean")
        stl_clean, _, _ = _export(mesh, name, "clean")
        
        # 2. Printed Version (with striae)
        mesh_printed = apply_layer_stratification(mesh, layer_height_cm=var.get('layer_height', 0.02))
        _, obj_printed, glb_printed = _export(mesh_printed, name, "printed")
        
        reports.append({
            "name": name,
            "val": val,
            "stl_clean": stl_clean,
            "obj_printed": obj_printed,
            "glb_printed": glb_printed
        })

    # Summary table
    print("\n" + "=" * 60)
    print("PRINTABILITY REPORT")
    print("=" * 60)
    for r in reports:
        v = r["val"]
        m = "✅" if v["manifold"] else "❌"
        t = "✅" if v["thickness_ok"] else "❌"
        o = "✅" if v["overhangs_ok"] else "⚠️"
        b = "✅" if v["flat_base"] else "❌"
        print(f"  {r['name']:<20} M:{m} T:{t} O:{o} B:{b}")
        
    return reports


if __name__ == "__main__":
    generate_all()
