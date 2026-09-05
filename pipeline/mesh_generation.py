"""
3D Mesh Generation Module.
Reconstructs watertight surface meshes from dense point clouds using Screened Poisson
Surface Reconstruction and Ball-Pivoting Algorithms (BPA) via Open3D.
"""
import numpy as np
import open3d as o3d
import trimesh
from pathlib import Path
from typing import Dict, Any, Optional
import config

class MeshGenerator:
    def __init__(self, models_dir: Path = config.MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def generate_mesh(self,
                      point_cloud_path: Optional[str] = None,
                      method: str = "poisson",
                      poisson_depth: int = config.RECONSTRUCTION["poisson_depth"]) -> Dict[str, Any]:
        """
        Generate a 3D surface mesh from the dense point cloud.
        Supports:
        - Screened Poisson Surface Reconstruction
        - Ball-Pivoting Algorithm (BPA)
        """
        if not point_cloud_path:
            point_cloud_path = str(self.models_dir / "dense_points_clean.ply")
            if not Path(point_cloud_path).exists():
                point_cloud_path = str(self.models_dir / "dense_points.ply")

        pcd_path_obj = Path(point_cloud_path)
        if not pcd_path_obj.exists():
            raise FileNotFoundError(f"Point cloud file not found for mesh generation: {point_cloud_path}")

        pcd = o3d.io.read_point_cloud(str(pcd_path_obj))
        if len(pcd.points) < 4:
            raise ValueError(f"Point cloud has insufficient points ({len(pcd.points)}) for triangulation.")

        # Compute adaptive nearest neighbor distance for robust geometry estimation
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = float(np.mean(distances)) if len(distances) > 0 else 0.05
        avg_dist = max(avg_dist, 1e-4)

        # Ensure normals are computed and consistently oriented with Open3D
        if not pcd.has_normals() or np.asarray(pcd.normals).shape[0] == 0:
            try:
                search_radius = max(avg_dist * 3.0, 0.05)
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=35)
                )
                try:
                    pcd.orient_normals_consistent_tangent_plane(k=15)
                except Exception:
                    pcd.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 5.0]))
            except Exception:
                pass

        mesh = None

        if method.lower() == "poisson":
            # Screened Poisson Reconstruction via Open3D
            mesh_raw, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=poisson_depth, width=0, scale=1.1, linear_fit=False
            )
            # Density trimming to remove spurious boundary bounding bubbles
            densities = np.asarray(densities)
            if len(densities) > 0:
                density_threshold = np.quantile(densities, 0.05)
                vertices_to_remove = densities < density_threshold
                mesh_raw.remove_vertices_by_mask(vertices_to_remove)
            mesh = mesh_raw

        elif method.lower() == "bpa":
            # Ball-Pivoting Algorithm (BPA) via Open3D
            radii = [avg_dist * 1.5, avg_dist * 3.0, avg_dist * 6.0]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii)
            )

        elif method.lower() in ("alpha", "alpha_shape"):
            # Alpha Shapes reconstruction via Open3D
            alpha = avg_dist * 3.5
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)

        if mesh is None or len(mesh.triangles) == 0:
            # Fallback to adaptive BPA if primary method yielded no surface
            radii = [avg_dist * 2.0, avg_dist * 4.0, avg_dist * 8.0]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii)
            )

        # Mesh cleanup: remove degenerate triangles and unreferenced vertices
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()

        # Surface smoothing: Taubin smoothing removes high-frequency photogrammetry noise without shrinkage
        if len(mesh.vertices) > 20 and len(mesh.triangles) > 20:
            try:
                mesh = mesh.filter_smooth_taubin(number_of_iterations=10, nu=-0.53, mu=0.5)
            except Exception:
                pass

        # Quadric Decimation: Optimize dense meshes for smooth 60fps WebGL/Three.js rendering
        max_triangles = 150000
        if len(mesh.triangles) > max_triangles:
            try:
                mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=max_triangles)
                mesh.remove_degenerate_triangles()
                mesh.remove_duplicated_triangles()
                mesh.remove_non_manifold_edges()
            except Exception:
                pass

        # Transfer point cloud colors to mesh vertices if point cloud has colors
        if pcd.has_colors() and len(pcd.colors) > 0 and len(mesh.vertices) > 0:
            try:
                from scipy.spatial import cKDTree
                pcd_pts = np.asarray(pcd.points)
                pcd_cols = np.asarray(pcd.colors)
                mesh_verts = np.asarray(mesh.vertices)
                tree = cKDTree(pcd_pts)
                _, idxs = tree.query(mesh_verts, k=1, workers=-1)
                mesh.vertex_colors = o3d.utility.Vector3dVector(pcd_cols[idxs])
            except Exception:
                pass

        mesh.compute_vertex_normals()

        # Save Mesh in PLY and OBJ formats
        mesh_ply_path = self.models_dir / "reconstructed_mesh.ply"
        mesh_obj_path = self.models_dir / "reconstructed_mesh.obj"

        o3d.io.write_triangle_mesh(str(mesh_ply_path), mesh)
        o3d.io.write_triangle_mesh(str(mesh_obj_path), mesh)

        return {
            "status": "SUCCESS",
            "method_used": method.upper(),
            "vertices_count": len(mesh.vertices),
            "faces_count": len(mesh.triangles),
            "has_vertex_colors": mesh.has_vertex_colors(),
            "has_normals": mesh.has_vertex_normals(),
            "mesh_ply": str(mesh_ply_path.resolve()),
            "mesh_obj": str(mesh_obj_path.resolve())
        }
