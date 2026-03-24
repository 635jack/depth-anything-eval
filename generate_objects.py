#!/usr/bin/env python3
"""
generate_objects.py — Génération de 6 objets 3D paramétriques pour évaluation DA-V2.

Chaque objet est vérifié (manifold, épaisseur, overhangs, base plate)
puis exporté en STL (impression) et OBJ (rendu Blender).
"""

import os
import numpy as np
import trimesh
from trimesh.creation import revolve
from shapely.geometry import Polygon


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "meshes")
MIN_THICKNESS_MM = 1.0        # épaisseur minimum acceptable (mm)
MAX_OVERHANG_ANGLE_DEG = 45   # angle max sans support


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
    """Vérifie que la dimension la plus petite du bounding-box >= min_mm."""
    extents = mesh.bounding_box.extents  # en mètres (on travaille en cm→m plus tard)
    # On travaille directement en cm dans trimesh, conversion en mm pour le check
    min_extent_mm = min(extents) * 10  # cm → mm
    ok = min_extent_mm >= min_mm
    status = f"✅ épaisseur min {min_extent_mm:.1f}mm" if ok else f"⚠️  épaisseur {min_extent_mm:.1f}mm < {min_mm}mm"
    print(f"  [{name}] {status}")
    return ok


def _check_overhangs(mesh: trimesh.Trimesh, name: str, max_angle_deg: float = MAX_OVERHANG_ANGLE_DEG) -> dict:
    """Détecte les faces avec un overhang > max_angle_deg par rapport à la verticale."""
    normals = mesh.face_normals
    # Angle entre la normale de la face et le vecteur "vers le haut" (Z+)
    up = np.array([0, 0, 1])
    cos_angles = np.dot(normals, up)
    angles_deg = np.degrees(np.arccos(np.clip(cos_angles, -1, 1)))
    # Les faces orientées vers le bas (angle > 90°) avec angle > (180 - max_angle)
    overhang_mask = angles_deg > (180 - max_angle_deg)
    n_overhang = int(overhang_mask.sum())
    pct = 100 * n_overhang / len(normals) if len(normals) > 0 else 0
    ok = pct < 5  # < 5% de faces en overhang = OK
    status = f"✅ overhangs {pct:.1f}%" if ok else f"⚠️  overhangs {pct:.1f}%"
    print(f"  [{name}] {status} ({n_overhang}/{len(normals)} faces)")
    return {"ok": ok, "pct": pct, "n_faces": n_overhang}


def _check_flat_base(mesh: trimesh.Trimesh, name: str) -> bool:
    """Vérifie qu'il y a des faces à Z ~= z_min (base plate)."""
    z_min = mesh.bounds[0, 2]
    z_tol = 0.01  # 0.01 cm tolérance
    face_centroids_z = mesh.triangles_center[:, 2]
    n_base = int(np.sum(np.abs(face_centroids_z - z_min) < z_tol))
    ok = n_base > 0
    status = f"✅ base plate ({n_base} faces)" if ok else "⚠️  pas de base plate"
    print(f"  [{name}] {status}")
    return ok


def _export(mesh: trimesh.Trimesh, name: str, out_dir: str = OUTPUT_DIR):
    """Exporte en STL, OBJ et GLB."""
    stl_path = os.path.join(out_dir, f"{name}.stl")
    obj_path = os.path.join(out_dir, f"{name}.obj")
    glb_path = os.path.join(out_dir, f"{name}.glb")
    mesh.export(stl_path)
    mesh.export(obj_path)
    mesh.export(glb_path)
    print(f"  [{name}] → {stl_path}")
    print(f"  [{name}] → {obj_path}")
    print(f"  [{name}] → {glb_path}")
    return stl_path, obj_path, glb_path


def _validate_and_export(mesh: trimesh.Trimesh, name: str) -> dict:
    """Valide et exporte un mesh, retourne le rapport."""
    print(f"\n{'='*50}")
    print(f"Objet : {name}")
    print(f"  Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")
    print(f"  Bounding box: {mesh.bounding_box.extents} cm")
    print(f"{'='*50}")

    manifold = _check_manifold(mesh, name)
    thickness = _check_thickness(mesh, name)
    overhangs = _check_overhangs(mesh, name)
    flat_base = _check_flat_base(mesh, name)
    # Export
    stl, obj, glb = _export(mesh, name)

    return {
        "name": name,
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "extents_cm": mesh.bounding_box.extents.tolist(),
        "manifold": manifold,
        "thickness_ok": thickness,
        "overhangs_pct": overhangs["pct"],
        "overhangs_ok": overhangs["ok"],
        "flat_base": flat_base,
        "stl_path": stl,
        "obj_path": obj,
    }


# ---------------------------------------------------------------------------
# Objet generators
# ---------------------------------------------------------------------------

def generate_ellipsoidal_cup(r_base, r_top, height, bulge=0.0, wall=0.003, segments=64):
    """
    Génère une tasse avec profil ellipsoïdal par révolution.
    """
    t = np.linspace(0, 1, 60)
    
    # Profil extérieur
    r_mid = (r_base + r_top) / 2 + bulge
    r_profile = (1 - t)**2 * r_base + 2*(1-t)*t * r_mid + t**2 * r_top
    z_profile = t * height
    
    # Profil intérieur (s'arrête à z=wall pour avoir un fond)
    t_inner = np.linspace(wall/height, 1, 60)
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

OBJECT_GENERATORS = {
    "cup_cylindre": lambda: generate_ellipsoidal_cup(0.04, 0.04, 0.08, 0.00),
    "cup_tonneau": lambda: generate_ellipsoidal_cup(0.04, 0.04, 0.08, +0.015),
    "cup_pince": lambda: generate_ellipsoidal_cup(0.04, 0.04, 0.08, -0.010),
    "cup_evase": lambda: generate_ellipsoidal_cup(0.03, 0.05, 0.08, +0.010),
    "cup_inverse": lambda: generate_ellipsoidal_cup(0.05, 0.03, 0.10, -0.005),
    "cup_bombe": lambda: generate_ellipsoidal_cup(0.04, 0.04, 0.05, +0.020),
}


def generate_all() -> list[dict]:
    """Génère tous les objets, les valide et les exporte. Retourne les rapports."""
    _ensure_output_dir()
    reports = []

    for name, gen_fn in OBJECT_GENERATORS.items():
        mesh = gen_fn()
        report = _validate_and_export(mesh, name)
        reports.append(report)

    # --- Rapport récapitulatif ---
    print("\n" + "=" * 60)
    print("RAPPORT D'IMPRIMABILITÉ")
    print("=" * 60)
    print(f"{'Objet':<20} {'Manifold':<10} {'Épaisseur':<12} {'Overhangs':<12} {'Base plate':<10}")
    print("-" * 64)
    for r in reports:
        m = "✅" if r["manifold"] else "❌"
        t = "✅" if r["thickness_ok"] else "❌"
        o = f"{'✅' if r['overhangs_ok'] else '⚠️'} {r['overhangs_pct']:.1f}%"
        b = "✅" if r["flat_base"] else "❌"
        print(f"  {r['name']:<18} {m:<10} {t:<12} {o:<12} {b:<10}")

    all_ok = all(
        r["manifold"] and r["thickness_ok"] and r["overhangs_ok"] and r["flat_base"]
        for r in reports
    )
    print(f"\n{'✅ Tous les objets sont imprimables !' if all_ok else '⚠️  Certains objets nécessitent une vérification.'}")
    print(f"Fichiers exportés dans : {OUTPUT_DIR}\n")

    return reports


if __name__ == "__main__":
    generate_all()
