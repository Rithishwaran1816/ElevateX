"""
ElevateX Open3D Native Desktop 3D Viewer.
Inspects reconstructed point clouds, surface meshes, and camera geometry using Open3D's OpenGL engine.
"""
import sys
import argparse
from pathlib import Path
import open3d as o3d

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
import config

def view_with_open3d(model_path: str = None):
    print("=" * 60)
    print("   ELEVATEX OPEN3D INTERACTIVE 3D VIEWER")
    print("=" * 60)

    # Auto-detect latest reconstructed 3D artifacts if no path provided
    if not model_path:
        candidates = [
            config.MODELS_DIR / "textured_model.ply",
            config.MODELS_DIR / "reconstructed_mesh.ply",
            config.MODELS_DIR / "dense_points_clean.ply",
            config.MODELS_DIR / "dense_points.ply",
            config.MODELS_DIR / "sparse_points.ply",
        ]
        for c in candidates:
            if c.exists():
                model_path = str(c)
                break

    if not model_path or not Path(model_path).exists():
        print("[ERROR] No 3D model found. Run the reconstruction pipeline first!")
        print(f"Looked in: {config.MODELS_DIR}")
        return

    path_obj = Path(model_path)
    print(f"\n[OPEN3D] Loading: {path_obj.name} ({path_obj.stat().st_size / 1024:.1f} KB)")

    geometries = []
    # Add coordinate frame for spatial orientation
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    geometries.append(coord_frame)

    if path_obj.suffix.lower() in [".ply", ".obj"]:
        # Try loading as triangle mesh first
        mesh = o3d.io.read_triangle_mesh(str(path_obj))
        if len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
            if not mesh.has_vertex_normals():
                mesh.compute_vertex_normals()
            print(f"[OPEN3D] Geometry type: Triangle Mesh")
            print(f"         Vertices: {len(mesh.vertices):,}")
            print(f"         Triangles: {len(mesh.triangles):,}")
            print(f"         Vertex Colors: {mesh.has_vertex_colors()}")
            geometries.append(mesh)
        else:
            # Fall back to point cloud
            pcd = o3d.io.read_point_cloud(str(path_obj))
            print(f"[OPEN3D] Geometry type: Point Cloud")
            print(f"         Points: {len(pcd.points):,}")
            print(f"         Colors: {pcd.has_colors()}")
            geometries.append(pcd)

    print("\n[CONTROLS]")
    print(" - Left Click + Drag: Rotate")
    print(" - Right Click + Drag / Shift + Left Drag: Pan")
    print(" - Scroll Wheel: Zoom")
    print(" - Press 'W': Toggle Wireframe Mode")
    print(" - Press 'N': Toggle Point Normals")
    print(" - Press 'Q' or 'ESC': Exit Viewer")
    print("-" * 60)

    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"ElevateX 3D Viewer - {path_obj.name}",
        width=1280,
        height=720,
        left=100,
        top=100
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View reconstructed 3D models with Open3D")
    parser.add_argument("--model", type=str, default=None, help="Path to PLY or OBJ model")
    args = parser.parse_args()
    view_with_open3d(args.model)
