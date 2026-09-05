"""
Stage 2 Runner: Frame Quality Assessment, Duplicate Filtering & Preprocessing.
"""
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.frame_quality import FrameQualityAssessor
from pipeline.image_preprocessing import ImagePreprocessor

def run_stage_2(blur_threshold: float = 80.0):
    print(f"\n==========================================")
    print(f"   ELEVATEX STAGE 2: QUALITY FILTERING")
    print(f"==========================================")

    assessor = FrameQualityAssessor()
    quality_res = assessor.assess_and_filter(blur_threshold=blur_threshold)

    print(f"[OK] Total Extracted: {quality_res['total_frames']}")
    print(f"[OK] Accepted Frames: {quality_res['accepted_frames']}")
    print(f"[OK] Rejected (Blur): {quality_res['rejected_blur']} | (Exposure): {quality_res['rejected_exposure']} | (Duplicates): {quality_res['rejected_duplicates']}")
    print(f"[OK] Average Sharpness: {quality_res['average_sharpness']}")

    print("\nApplying Image Preprocessing (CLAHE Illumination Normalization)...")
    preprocessor = ImagePreprocessor()
    prep_res = preprocessor.preprocess_images()
    print(f"[OK] Preprocessed {prep_res['preprocessed_count']} clean frames into {prep_res['output_directory']}")

    return True, {"quality_meta": quality_res, "preprocessing_meta": prep_res}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ElevateX Stage 2: Quality Filtering & Preprocessing")
    parser.add_argument("--blur_threshold", type=float, default=80.0, help="Laplacian variance blur threshold")
    args = parser.parse_args()

    success, _ = run_stage_2(args.blur_threshold)
    sys.exit(0 if success else 1)
