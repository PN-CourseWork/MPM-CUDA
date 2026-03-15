#!/usr/bin/env python
"""Pipeline: H5 → splashsurf surface reconstruction → OBJ sequence → Blender render → MP4.

Usage:
    uv run python render_blender.py output/snowball.h5
    uv run python render_blender.py output/snowball.h5 --resolution 1920 1080 --samples 64
"""

from __future__ import annotations

import argparse
import subprocess
import textwrap
from pathlib import Path

import h5py
import numpy as np
import pysplashsurf


# ── Step 1: Surface reconstruction ──────────────────────────────────────────

def reconstruct_frames(h5_path: Path, mesh_dir: Path) -> int:
    """Read H5, run splashsurf per frame, write OBJ meshes."""
    mesh_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        positions = f["positions"][:]  # (n_frames, N, 3)

    n_frames = positions.shape[0]
    print(f"Reconstructing {n_frames} frames ({positions.shape[1]} particles) …")

    # Estimate particle radius from nearest-neighbor distance
    from scipy.spatial import cKDTree

    pts0 = positions[0].astype(np.float64)
    tree = cKDTree(pts0)
    dd, _ = tree.query(pts0, k=2)
    particle_radius = float(np.median(dd[:, 1]) / 2.0)
    print(f"  particle_radius = {particle_radius:.5f}")

    for i in range(n_frames):
        pts = positions[i].astype(np.float64)

        mesh_data, _ = pysplashsurf.reconstruction_pipeline(
            pts,
            particle_radius=particle_radius,
            rest_density=1000.0,
            smoothing_length=2.0,
            cube_size=1.0,
            iso_surface_threshold=0.6,
            compute_normals=True,
            normals_smoothing_iters=10,
            mesh_smoothing_iters=15,
            mesh_smoothing_weights=True,
            mesh_cleanup=True,
        )

        out_path = mesh_dir / f"frame_{i:04d}.obj"
        mesh_data.write_to_file(str(out_path), file_format="obj")

        if (i + 1) % 5 == 0 or i == n_frames - 1:
            print(f"  {i + 1}/{n_frames} → {mesh_data.nvertices} verts, {mesh_data.ncells} tris")

    print(f"Meshes saved to {mesh_dir}/")
    return n_frames


# ── Step 2: Blender render ──────────────────────────────────────────────────

def render_in_blender(
    mesh_dir: Path,
    output_path: Path,
    n_frames: int,
    resolution: tuple[int, int] = (1920, 1080),
    samples: int = 32,
    fps: int = 30,
) -> None:
    """Launch Blender in background mode to render the OBJ sequence to MP4."""
    frame_dir = str(mesh_dir.resolve())
    render_dir = str((output_path.parent / "blender_frames").resolve())

    blender_script = textwrap.dedent(f"""\
        import bpy
        import os
        import glob
        import mathutils

        # ── Clean scene ──
        bpy.ops.wm.read_factory_settings(use_empty=True)

        scene = bpy.context.scene
        scene.render.resolution_x = {resolution[0]}
        scene.render.resolution_y = {resolution[1]}
        scene.render.engine = 'CYCLES'
        scene.cycles.samples = {samples}
        scene.cycles.device = 'GPU'

        # Enable Metal GPU on macOS
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'METAL'
        prefs.get_devices()
        for dev in prefs.devices:
            dev.use = True
            print(f"  Device: {{dev.name}} ({{dev.type}}) use={{dev.use}}")
        scene.frame_start = 0
        scene.frame_end = {n_frames - 1}
        scene.render.fps = {fps}

        # ── World background ──
        world = bpy.data.worlds.new("World")
        scene.world = world
        bg = world.node_tree.nodes["Background"]
        bg.inputs["Color"].default_value = (0.02, 0.02, 0.05, 1.0)
        bg.inputs["Strength"].default_value = 0.5

        # ── Snow material ──
        mat = bpy.data.materials.new("Snow")
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        output_node = nodes.new("ShaderNodeOutputMaterial")
        principled = nodes.new("ShaderNodeBsdfPrincipled")
        principled.inputs["Base Color"].default_value = (0.92, 0.93, 1.0, 1.0)
        principled.inputs["Roughness"].default_value = 0.7
        principled.inputs["Subsurface Weight"].default_value = 0.4
        principled.inputs["Subsurface Radius"].default_value = (0.01, 0.01, 0.02)
        links.new(principled.outputs["BSDF"], output_node.inputs["Surface"])

        # ── Load first frame to get bounding box ──
        mesh_dir = r"{frame_dir}"
        obj_files = sorted(glob.glob(os.path.join(mesh_dir, "frame_*.obj")))
        n = len(obj_files)
        print(f"Found {{n}} OBJ files in {{mesh_dir}}")

        bpy.ops.wm.obj_import(filepath=obj_files[0])
        snow_obj = bpy.context.selected_objects[0]
        snow_obj.name = "SnowMesh"
        snow_obj.data.materials.append(mat)
        for poly in snow_obj.data.polygons:
            poly.use_smooth = True

        # Compute bounding box
        bbox = [snow_obj.matrix_world @ mathutils.Vector(c) for c in snow_obj.bound_box]
        bb_min = mathutils.Vector((min(v.x for v in bbox), min(v.y for v in bbox), min(v.z for v in bbox)))
        bb_max = mathutils.Vector((max(v.x for v in bbox), max(v.y for v in bbox), max(v.z for v in bbox)))
        bb_center = (bb_min + bb_max) / 2
        bb_size = bb_max - bb_min
        max_dim = max(bb_size.x, bb_size.y, bb_size.z)
        print(f"Mesh BB center: {{bb_center}}, size: {{bb_size}}")

        # ── Ground plane ──
        ground_z = bb_min.z - 0.005
        bpy.ops.mesh.primitive_plane_add(size=2.0, location=(bb_center.x, bb_center.y, ground_z))
        ground = bpy.context.active_object
        ground_mat = bpy.data.materials.new("Ground")
        gn = ground_mat.node_tree.nodes
        gl = ground_mat.node_tree.links
        gn.clear()
        g_out = gn.new("ShaderNodeOutputMaterial")
        g_bsdf = gn.new("ShaderNodeBsdfPrincipled")
        g_bsdf.inputs["Base Color"].default_value = (0.08, 0.08, 0.12, 1.0)
        g_bsdf.inputs["Roughness"].default_value = 0.95
        gl.new(g_bsdf.outputs["BSDF"], g_out.inputs["Surface"])
        ground.data.materials.append(ground_mat)

        # ── Lighting ──
        bpy.ops.object.light_add(type='SUN', location=(2, 2, 3))
        sun = bpy.context.active_object
        sun.data.energy = 3.0
        sun.rotation_euler = (0.6, 0.1, -0.3)

        bpy.ops.object.light_add(type='AREA', location=(bb_center.x - 0.5, bb_center.y - 0.3, bb_center.z + 0.5))
        fill = bpy.context.active_object
        fill.data.energy = 30.0
        fill.data.size = 0.5

        # ── Camera ──
        cam_dist = max_dim * 2.5
        cam_loc = (bb_center.x, bb_center.y - cam_dist, bb_center.z + max_dim * 0.5)
        bpy.ops.object.camera_add(location=cam_loc)
        cam = bpy.context.active_object
        scene.camera = cam
        cam.data.lens = 50

        bpy.ops.object.empty_add(location=bb_center)
        target = bpy.context.active_object
        target.name = "CameraTarget"
        track = cam.constraints.new('TRACK_TO')
        track.target = target
        track.track_axis = 'TRACK_NEGATIVE_Z'
        track.up_axis = 'UP_Y'

        # ── Mesh sequence: cache and swap on frame change ──
        mesh_cache = {{}}
        mesh_cache[0] = snow_obj.data.copy()

        def load_frame(scene, depsgraph=None):
            frame = scene.frame_current
            if frame < 0 or frame >= n:
                return
            if frame not in mesh_cache:
                bpy.ops.wm.obj_import(filepath=obj_files[frame])
                imported = bpy.context.selected_objects[0]
                mesh_cache[frame] = imported.data.copy()
                bpy.data.objects.remove(imported)
            snow_obj.data = mesh_cache[frame]
            snow_obj.data.materials.clear()
            snow_obj.data.materials.append(mat)
            for poly in snow_obj.data.polygons:
                poly.use_smooth = True

        bpy.app.handlers.frame_change_pre.append(load_frame)

        # ── Render animation ──
        render_dir = r"{render_dir}"
        os.makedirs(render_dir, exist_ok=True)
        scene.render.filepath = os.path.join(render_dir, "frame_")
        scene.render.image_settings.file_format = 'PNG'

        print(f"Rendering {{n}} frames to {{render_dir}} ...")
        bpy.ops.render.render(animation=True)
        print("Render complete.")
    """)

    script_path = mesh_dir.parent / "blender_render.py"
    script_path.write_text(blender_script)

    print(f"Launching Blender to render {n_frames} frames …")
    subprocess.run(
        ["blender", "--background", "--python", str(script_path)],
        check=True,
    )

    # Combine PNGs to MP4 with ffmpeg
    render_dir_path = output_path.parent / "blender_frames"
    print("Encoding MP4 with ffmpeg …")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(render_dir_path / "frame_%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "18",
            str(output_path),
        ],
        check=True,
    )
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved: {output_path} ({size_mb:.1f} MB)")


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="H5 → splashsurf → Blender → MP4")
    parser.add_argument("h5_path", type=Path, help="Path to simulation .h5 file")
    parser.add_argument("--output", type=Path, default=None, help="Output MP4 path")
    parser.add_argument("--resolution", type=int, nargs=2, default=[1920, 1080])
    parser.add_argument("--samples", type=int, default=32, help="Cycles render samples")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--skip-reconstruct", action="store_true", help="Skip surface reconstruction (use existing OBJs)")
    args = parser.parse_args()

    h5 = args.h5_path
    out_mp4 = args.output or h5.with_name(h5.stem + "_blender.mp4")
    mesh_dir = h5.with_suffix("") / "meshes"

    if args.skip_reconstruct:
        n_frames = len(list(mesh_dir.glob("frame_*.obj")))
        print(f"Skipping reconstruction, found {n_frames} existing OBJ files")
    else:
        n_frames = reconstruct_frames(h5, mesh_dir)

    render_in_blender(
        mesh_dir, out_mp4, n_frames,
        resolution=tuple(args.resolution),
        samples=args.samples,
        fps=args.fps,
    )
