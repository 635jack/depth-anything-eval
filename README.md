# Depth Anything V2 — Evaluation Pipeline

Pipeline d'évaluation de **Depth Anything V2** sur des objets 3D paramétriques rendus dans Blender.

## Workflow

1. **Génération** — 6 objets 3D paramétriques (STL + OBJ) via `trimesh`
2. **Rendu** — RGB (PNG) + depth ground-truth (EXR 32-bit, mètres réels) via Blender CLI
3. **Inférence** — Depth Anything V2 Large (MPS / CPU fallback)
4. **Métriques** — AbsRel, SqRel, RMSE, RMSElog, δ1.25 / δ1.25² / δ1.25³
5. **Rapport** — Figures comparatives + rapport HTML + CSV

## Prérequis

- Python ≥ 3.10
- Blender ≥ 3.6 (accessible en CLI)
- GPU MPS (Apple Silicon) ou CPU

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

```bash
# Pipeline complet
python main.py

# Modules individuels
python generate_objects.py
blender --background --python render_scene.py -- --obj_dir output/meshes --output_dir output/renders
python run_da2.py
python metrics.py
python visualize.py
```

## Objets générés

| Objet | Description | Dimensions |
|-------|-------------|------------|
| `box_simple` | Cube simple | 8×8×8 cm |
| `stepped_box` | Boîte à 2 niveaux | 10×10×6 cm |
| `cylinder_flat` | Cylindre plat | Ø8 cm, H=4 cm |
| `l_shape` | Forme en L extrudée | 12×8×4 cm |
| `t_shape` | Forme en T extrudée | 10×8×4 cm |
| `pyramid_stepped` | Pyramide étagée 3 niveaux | variable |

## Structure

```
├── generate_objects.py   # Génération des meshes
├── render_scene.py       # Script Blender CLI
├── run_da2.py            # Inférence DA-V2
├── metrics.py            # Calcul des métriques
├── visualize.py          # Visualisation + rapport HTML
├── main.py               # Orchestrateur
└── output/               # Données générées (gitignored)
```
