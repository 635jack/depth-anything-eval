import bpy
import os
import sys
import math
import argparse
import mathutils

# Parse args passed after '--'
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = []

parser = argparse.ArgumentParser()
parser.add_argument("--obj_dir", required=True)
parser.add_argument("--output_dir", required=True)
args = parser.parse_args(argv)

os.makedirs(args.output_dir, exist_ok=True)

# Delete all default objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

scene = bpy.context.scene

# Setup render engine
scene.render.engine = 'CYCLES'
try:
    scene.cycles.device = 'GPU'
except:
    pass # fallback to CPU
scene.cycles.samples = 64
scene.render.resolution_x = 640
scene.render.resolution_y = 480

# Setup Compositor for Depth EXR
scene.use_nodes = True
tree = scene.node_tree
links = tree.links

for n in tree.nodes:
    tree.nodes.remove(n)

rl = tree.nodes.new(type="CompositorNodeRLayers")
comp = tree.nodes.new(type="CompositorNodeComposite")
links.new(rl.outputs['Image'], comp.inputs['Image'])

depth_output = tree.nodes.new(type="CompositorNodeOutputFile")
depth_output.format.file_format = 'OPEN_EXR'
depth_output.format.color_depth = '32'
depth_output.format.color_management = 'OVERRIDE' # Raw data
bpy.context.view_layer.use_pass_z = True
print("Render Layer Outputs:", list(rl.outputs.keys()))
depth_out_name = 'Depth' if 'Depth' in rl.outputs else 'Z'
links.new(rl.outputs[depth_out_name], depth_output.inputs[0])

# Background
world = bpy.context.scene.world
if world is None:
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
world.use_nodes = True
bg_node = world.node_tree.nodes.get("Background")
if bg_node:
    bg_node.inputs[0].default_value = (0.5, 0.5, 0.5, 1.0) # Light gray
    bg_node.inputs[1].default_value = 1.0

# Camera
cam_data = bpy.data.cameras.new("Camera")
cam_obj = bpy.data.objects.new("Camera", cam_data)
bpy.context.collection.objects.link(cam_obj)
scene.camera = cam_obj

# Setup 3-point lighting
def look_at(obj, target):
    direction = target - obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    obj.rotation_euler = rot_quat.to_euler()

def add_light(name, pos, energy, type='AREA', size=0.2):
    light_data = bpy.data.lights.new(name=name, type=type)
    light_data.energy = energy
    if type == 'AREA':
        light_data.size = size
    light_obj = bpy.data.objects.new(name=name, object_data=light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = pos
    look_at(light_obj, mathutils.Vector((0,0,0)))
    return light_obj

# Lights
key_light = add_light("Key", (-0.5, -0.5, 0.5), 50.0)
fill_light = add_light("Fill", (0.5, -0.3, 0.1), 20.0)
back_light = add_light("Back", (0, 0.5, 0.5), 30.0)

# Materials
def setup_matte_material(obj, color=(0.1, 0.4, 0.8, 1.0)):
    mat = bpy.data.materials.new(name="Matte")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs['Base Color'].default_value = color
    bsdf.inputs['Roughness'].default_value = 0.85
    bsdf.inputs['Specular IOR Level'].default_value = 0.1
    obj.data.materials.append(mat)

def setup_silk_material(obj, color=(0.8, 0.2, 0.1, 1.0)):
    mat = bpy.data.materials.new(name="Silk")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs['Base Color'].default_value = color
    bsdf.inputs['Roughness'].default_value = 0.2
    bsdf.inputs['Metallic'].default_value = 0.3
    bsdf.inputs['Coat Weight'].default_value = 0.5
    bsdf.inputs['Anisotropic'].default_value = 0.7
    obj.data.materials.append(mat)

def setup_petg_material(obj, color=(0.1, 0.8, 0.4, 1.0)):
    mat = bpy.data.materials.new(name="PETG")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs['Base Color'].default_value = color
    bsdf.inputs['Transmission Weight'].default_value = 0.95
    bsdf.inputs['Roughness'].default_value = 0.2
    bsdf.inputs['IOR'].default_value = 1.54
    obj.data.materials.append(mat)

# Angles: 0° front, 45° side, 30° elevated top
angles = [
    {"name": "front", "pos": (0, -0.5, 0)},
    {"name": "side45", "pos": (0.35355, -0.35355, 0)},
    {"name": "top30", "pos": (0.35355, -0.35355, 0.288)} # ~30 deg elev
]

# Process models
for f in sorted(os.listdir(args.obj_dir)):
    if not f.endswith('_printed.obj'):
        continue
    
    obj_name = f[:-len('_printed.obj')]
    obj_path = os.path.join(args.obj_dir, f)
    
    # Import
    bpy.ops.wm.obj_import(filepath=obj_path)
    imported = bpy.context.selected_objects
    
    if not imported:
        print(f"Failed to import {f}")
        continue
        
    main_obj = imported[0]
    
    # Scale from cm to meters (1 cm = 0.01 m)
    main_obj.scale = (0.01, 0.01, 0.01)
    bpy.context.view_layer.update()
    
    # Material assignment
    if "matte" in obj_name.lower():
        setup_matte_material(main_obj, (0.05, 0.2, 0.6, 1.0)) # Dark Blue
    elif "silk" in obj_name.lower():
        setup_silk_material(main_obj, (0.7, 0.1, 0.1, 1.0)) # Silk Red
    elif "translucent" in obj_name.lower():
        setup_petg_material(main_obj, (0.1, 0.8, 0.4, 0.5)) # Translucent Green
    else:
        # Default fallback
        setup_matte_material(main_obj, (0.4, 0.4, 0.4, 1.0))
    
    # Ensure camera tracks the object center
    if not cam_obj.constraints:
        track = cam_obj.constraints.new(type='TRACK_TO')
        track.track_axis = 'TRACK_NEGATIVE_Z'
        track.up_axis = 'UP_Y'
    else:
        track = cam_obj.constraints[0]
    track.target = main_obj
    
    for angle in angles:
        # Distance = 50cm = 0.5m. We adjust Z of camera position to adapt to obj height
        z_offset = main_obj.dimensions[2] / 2
        cam_obj.location = (angle["pos"][0], angle["pos"][1], angle["pos"][2] + z_offset)
        
        # Base names
        rgb_path = os.path.join(args.output_dir, f"{obj_name}_{angle['name']}.png")
        scene.render.filepath = rgb_path
        
        depth_output.base_path = args.output_dir
        # Ex file_slots[0].path behaves as prefix for frames
        prefix = f"{obj_name}_{angle['name']}_depth_"
        depth_output.file_slots[0].path = prefix
        
        print(f"Rendering {obj_name} @ {angle['name']}")
        bpy.ops.render.render(write_still=True)
        
        # OpenEXR file is written as `<prefix>0001.exr`
        exr_path = os.path.join(args.output_dir, f"{prefix}0001.exr")
        final_exr_path = os.path.join(args.output_dir, f"{obj_name}_{angle['name']}.exr")
        if os.path.exists(exr_path):
            os.rename(exr_path, final_exr_path)
            
    # Cleanup object
    bpy.ops.object.select_all(action='DESELECT')
    for o in imported:
        o.select_set(True)
    bpy.ops.object.delete()

print("Rendering complete.")
