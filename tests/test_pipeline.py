"""
Integration Tests for ElevateX Pipeline.
"""
import pytest
import numpy as np
import open3d as o3d
from pathlib import Path
import config
from pipeline.mesh_generation import MeshGenerator
from pipeline.texture_generation import TextureGenerator
from pipeline.georeferencing import Georeferencer

def test_mesh_and_texture_generation(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Create synthetic point cloud with normals
    pts = []
    normals = []
    colors = []
    for x in np.linspace(-1, 1, 15):
        for y in np.linspace(-1, 1, 15):
            pts.append([x, y, 0.0])
            normals.append([0.0, 0.0, 1.0])
            colors.append([0.2, 0.8, 0.4])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(pts))
    pcd.normals = o3d.utility.Vector3dVector(np.array(normals))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

    pcd_path = models_dir / "dense_points_clean.ply"
    o3d.io.write_point_cloud(str(pcd_path), pcd)

    # Test Mesh Generation (BPA)
    mesh_gen = MeshGenerator(models_dir=models_dir)
    mesh_res = mesh_gen.generate_mesh(point_cloud_path=str(pcd_path), method="bpa")

    assert mesh_res["status"] == "SUCCESS"
    assert mesh_res["vertices_count"] > 0
    assert Path(mesh_res["mesh_ply"]).exists()

    # Test Texture Generation & Export
    tex_gen = TextureGenerator(models_dir=models_dir)
    tex_res = tex_gen.generate_textured_model(mesh_path=mesh_res["mesh_ply"], point_cloud_path=str(pcd_path))

    assert tex_res["status"] == "TEXTURE_APPLIED"
    assert Path(tex_res["textured_ply"]).exists()

def test_georeferencing_fallback(tmp_path):
    reports_dir = tmp_path / "reports"
    georef = Georeferencer(reports_dir=reports_dir)

    # Without GPS
    res_none = georef.process_georeferencing(telemetry_data=None)
    assert res_none["georeferencing_status"] == "LOCAL_COORDINATE_SYSTEM"
    assert "unavailable" in res_none["message"].lower()

    # With GPS
    fake_telemetry = [
        {"latitude": 28.6139, "longitude": 77.2090, "altitude": 100.0},
        {"latitude": 28.6140, "longitude": 77.2091, "altitude": 102.0}
    ]
    res_gps = georef.process_georeferencing(telemetry_data=fake_telemetry)
    assert res_gps["georeferencing_status"] == "GEOREFERENCED_WGS84_UTM"
    assert "utm_zone" in res_gps
    assert res_gps["anchor_wgs84"]["latitude"] == 28.61395
