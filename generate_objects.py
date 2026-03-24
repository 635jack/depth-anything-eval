#!/usr/bin/env python3
"""
generate_objects.py — Génération de 6 objets 3D paramétriques pour évaluation DA-V2.

Chaque objet est vérifié (manifold, épaisseur, overhangs, base plate)
puis exporté en STL (impression) et OBJ (rendu Blender).
"""

import os
import numpy as np
import trimesh
from trimesh.creation import extrude_polygon
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


def _export(mesh: trimesh.Trimesh, name: str):
    """Exporte en STL et OBJ."""
    stl_path = os.path.join(OUTPUT_DIR, f"{name}.stl")
    obj_path = os.path.join(OUTPUT_DIR, f"{name}.obj")
    mesh.export(stl_path, file_type="stl")
    mesh.export(obj_path, file_type="obj")
    print(f"  [{name}] → {stl_path}")
    print(f"  [{name}] → {obj_path}")
    return stl_path, obj_path


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
    stl, obj = _export(mesh, name)

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

def make_box_simple() -> trimesh.Trimesh:
    """Cube 8×8×8 cm."""
    mesh = trimesh.creation.box(extents=[8, 8, 8])
    # Translate pour que la base soit à Z=0
    mesh.apply_translation([0, 0, 4])
    return mesh


def make_stepped_box() -> trimesh.Trimesh:
    """Boîte avec marche (2 niveaux), 10×10×6 cm total.
    Niveau 1 : 10×10×3 cm (base)
    Niveau 2 : 5×10×3 cm  (marche supérieure, décalée)
    """
    base = trimesh.creation.box(extents=[10, 10, 3])
    base.apply_translation([0, 0, 1.5])

    step = trimesh.creation.box(extents=[5, 10, 3])
    step.apply_translation([2.5, 0, 4.5])

    mesh = trimesh.boolean.union([base, step], engine="blender")
    if not isinstance(mesh, trimesh.Trimesh):
        # Fallback : concaténation simple si le moteur booléen échoue
        mesh = trimesh.util.concatenate([base, step])
    return mesh


def make_stepped_box_fallback() -> trimesh.Trimesh:
    """Fallback pour stepped_box sans opérations booléennes.
    Crée la forme directement par extrusion de profil L en coupe latérale.
    """
    # Profil en coupe (vue de côté, plan XZ) — forme en L inversé
    # Base: 10cm largeur, 3cm haut
    # Marche: 5cm largeur (droite), 3cm haut supplémentaire
    profile = Polygon([
        (0, 0), (10, 0), (10, 3),
        (5, 3), (5, 6),
        (0, 6), (0, 0)
    ])
    # Extrude le long de Y sur 10 cm
    mesh = extrude_polygon(profile, height=10)
    # Repositionner : centrer XY, base à Z=0
    # L'extrusion met le profil dans le plan XY et extrude en Z par défaut
    # On doit réorienter : le profil est dans XZ, extrusion en Y
    # trimesh extrude_polygon extrude le long de Z, donc le profil doit être dans XY
    # Notre profil est déjà dans XY, mais représente la vue de côté.
    # Après extrusion, on a le profil dans X-Y extrudé en Z (profondeur).
    # On doit pivoter pour que Z soit la hauteur.

    # Pivoter -90° autour de X pour que l'extrusion (Z) devienne Y (profondeur)
    rot = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    mesh.apply_transform(rot)

    # Recentrer en XY, base à Z=0
    bounds = mesh.bounds
    center_xy = (bounds[0, :2] + bounds[1, :2]) / 2
    z_min = bounds[0, 2]
    mesh.apply_translation([-center_xy[0], -center_xy[1], -z_min])

    return mesh


def make_cylinder_flat() -> trimesh.Trimesh:
    """Cylindre plat D=8 cm, H=4 cm."""
    mesh = trimesh.creation.cylinder(radius=4, height=4, sections=64)
    mesh.apply_translation([0, 0, 2])
    return mesh


def make_l_shape() -> trimesh.Trimesh:
    """L-shape extrudé, 12×8×4 cm.
    Profil L dans le plan XY, extrudé en Z (hauteur = 4 cm).
    """
    # Profil en L (vue du dessus)
    # Branche longue : 12 cm en X, 3 cm en Y
    # Branche courte : 3 cm en X, 8 cm en Y total
    profile = Polygon([
        (0, 0), (12, 0), (12, 3),
        (3, 3), (3, 8), (0, 8), (0, 0)
    ])
    mesh = extrude_polygon(profile, height=4)
    # Centrer XY, base à Z=0
    bounds = mesh.bounds
    center_xy = (bounds[0, :2] + bounds[1, :2]) / 2
    mesh.apply_translation([-center_xy[0], -center_xy[1], 0])
    return mesh


def make_t_shape() -> trimesh.Trimesh:
    """T-shape extrudé, 10×8×4 cm.
    Profil T dans le plan XY, extrudé en Z (hauteur = 4 cm).
    """
    # Barre horizontale du T : 10 cm en X, 3 cm en Y (en haut)
    # Barre verticale du T : 3 cm en X, 5 cm en Y (en bas, centrée)
    profile = Polygon([
        (0, 5), (10, 5), (10, 8),
        (0, 8), (0, 5),  # barre haute
    ])
    stem = Polygon([
        (3.5, 0), (6.5, 0), (6.5, 5),
        (3.5, 5), (3.5, 0)
    ])
    # Union des deux polygones
    t_profile = profile.union(stem)
    mesh = extrude_polygon(t_profile, height=4)
    # Centrer XY, base à Z=0
    bounds = mesh.bounds
    center_xy = (bounds[0, :2] + bounds[1, :2]) / 2
    mesh.apply_translation([-center_xy[0], -center_xy[1], 0])
    return mesh


def make_pyramid_stepped() -> trimesh.Trimesh:
    """Pyramide étagée 3 niveaux.
    Niveau 1 : 10×10×2 cm
    Niveau 2 : 7×7×2 cm
    Niveau 3 : 4×4×2 cm
    """
    levels = [
        (10, 10, 2, 1),   # largeur, profondeur, hauteur, z_offset
        (7, 7, 2, 3),
        (4, 4, 2, 5),
    ]
    meshes = []
    for w, d, h, z in levels:
        box = trimesh.creation.box(extents=[w, d, h])
        box.apply_translation([0, 0, z])
        meshes.append(box)

    mesh = trimesh.util.concatenate(meshes)
    return mesh


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

OBJECT_GENERATORS = {
    "box_simple": make_box_simple,
    "stepped_box": make_stepped_box_fallback,  # utilise le fallback sans booléen
    "cylinder_flat": make_cylinder_flat,
    "l_shape": make_l_shape,
    "t_shape": make_t_shape,
    "pyramid_stepped": make_pyramid_stepped,
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
