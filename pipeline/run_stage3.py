"""
Stage 3 Runner: Structure from Motion (SfM) and Dense Multi-View Reconstruction.
"""
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.reconstruction import PhotogrammetryReconstructor

def run_stage_3(force_builtin: bool = False):
    print(f"\n==========================================")
    print(f"   ELEVATEX STAGE 3: 3D RECONSTRUCTION")
    print(f"==========================================")

    reconstructor = PhotogrammetryReconstructor()
    res = reconstructor.run_reconstruction(force_builtin=force_builtin)

    print(f"[OK] Reconstruction Engine: {res.get('engine')}")
    print(f"[OK] Sparse Point Cloud: {res.get('sparse_ply')}")
    print(f"[OK] Dense Point Cloud: {res.get('dense_ply')}")
    if "sparse_points_count" in res:
        print(f"[OK] Sparse Points: {res.get('sparse_points_count'):,} | Dense Points: {res.get('dense_points_count'):,}")

    return True, res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ElevateX Stage 3: SfM & Dense Reconstruction")
    parser.add_argument("--force_builtin", action="store_true", help="Force built-in OpenCV/Open3D SfM instead of COLMAP")
    args = parser.parse_args()

    success, _ = run_stage_3(args.force_builtin)
    sys.exit(0 if success else 1)
