"""
Stage 4 Runner: Mesh Generation, Texture Baking, Georeferencing & Quality Assessment.
"""
import sys
import time
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.mesh_generation import MeshGenerator
from pipeline.texture_generation import TextureGenerator
from pipeline.georeferencing import Georeferencer
from pipeline.quality_assessment import QualityAssessor
from services.metadata_service import MetadataService

def run_stage_4(telemetry_file: str = None, method: str = "poisson"):
    print(f"\n=======================================================")
    print(f"   ELEVATEX STAGE 4: MESH, TEXTURE & GEOREFERENCING")
    print(f"=======================================================")

    # 1. Mesh Generation
    print("Generating watertight 3D Surface Mesh (Poisson)...")
    mesh_gen = MeshGenerator()
    mesh_res = mesh_gen.generate_mesh(method=method)
    print(f"[OK] Mesh Generated: {mesh_res['vertices_count']:,} Vertices | {mesh_res['faces_count']:,} Triangles")
    print(f"[OK] Mesh Path: {mesh_res['mesh_ply']}")

    # 2. Texture Generation & GLB Export
    print("\nBaking vertex texture colors & exporting GLB/OBJ/PLY models...")
    tex_gen = TextureGenerator()
    tex_res = tex_gen.generate_textured_model(mesh_path=mesh_res['mesh_ply'])
    print(f"[OK] Formats Exported: {', '.join(tex_res['export_formats'])}")
    if tex_res.get('model_glb'):
        print(f"[OK] Browser GLB Model: {tex_res['model_glb']}")

    # 3. Georeferencing
    print("\nProcessing Georeferencing Layer...")
    telemetry_data = []
    sensor_status = MetadataService.get_sensor_status_summary(None)
    if telemetry_file and Path(telemetry_file).exists():
        parsed_tel = MetadataService.parse_flight_telemetry(telemetry_file)
        telemetry_data = parsed_tel.get("summary", {}).get("telemetry_data", [])
        sensor_status = MetadataService.get_sensor_status_summary(parsed_tel)

    georef = Georeferencer()
    georef_res = georef.process_georeferencing(telemetry_data=telemetry_data)
    print(f"[STATUS] {georef_res['georeferencing_status']}: {georef_res['message']}")

    return True, {
        "mesh_meta": mesh_res,
        "texture_meta": tex_res,
        "georeference_meta": georef_res,
        "sensor_status": sensor_status
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ElevateX Stage 4: Mesh, Texture & Georeferencing")
    parser.add_argument("--telemetry", type=str, default=None, help="Path to telemetry CSV/JSON/SRT")
    parser.add_argument("--method", type=str, default="poisson", help="Mesh algorithm: poisson or bpa")
    args = parser.parse_args()

    success, _ = run_stage_4(args.telemetry, args.method)
    sys.exit(0 if success else 1)
