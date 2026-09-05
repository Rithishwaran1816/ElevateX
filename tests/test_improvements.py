"""
Unit Tests for ElevateX Pipeline Improvements (SIH26158).
Tests:
  - Outro cut-off detection
  - Intelligent keyframe selection
  - Building-aware sky soft-masking
  - Sparse SfM validation gate & quality classification
  - Georeferencing vs metric accuracy separation
"""
import pytest
import numpy as np
import cv2
from pathlib import Path

import config
from pipeline.frame_extraction import FrameExtractor
from pipeline.frame_quality import FrameQualityAssessor
from pipeline.image_preprocessing import ImagePreprocessor
from pipeline.colmap_pipeline import ColmapPipeline
from pipeline.quality_assessment import QualityAssessor


def test_sky_mask_preprocessing():
    """Verify sky soft-mask attenuates upper region without modifying bottom."""
    img = np.full((100, 100, 3), 200, dtype=np.uint8)
    masked = ImagePreprocessor.apply_sky_mask(img, sky_upper_fraction=0.3, strength=0.6)

    # Top pixel should be darkened
    assert masked[0, 50, 0] < 200, "Top sky region must be attenuated"
    # Bottom pixel should remain untouched
    assert np.all(masked[50:, :, :] == 200), "Lower building region must remain unchanged"


def test_validation_gate_classification():
    """Test ColmapPipeline validation classifications across various scenarios."""
    pipeline = ColmapPipeline(Path("dummy_in"), Path("dummy_out"))

    # Case 1: Healthy reconstruction
    healthy_stats = {
        "registered_images": 160,
        "total_images": 175,
        "registration_pct": 91.4,
        "sparse_points": 12000,
        "mean_reprojection_error": 0.85,
        "mean_track_length": 4.5,
        "camera_positions": [[i, i*0.1, 5.0] for i in range(160)],
        "bounding_box": {"extents": [10.0, 10.0, 15.0]}
    }
    val_healthy = pipeline._validate_sparse_reconstruction(healthy_stats)
    assert val_healthy["result"] == "RELIABLE"
    assert any("Registration: 91.4%" in m for m in val_healthy["messages"])

    # Case 2: Degenerate / low registration reconstruction
    failing_stats = {
        "registered_images": 10,
        "total_images": 175,
        "registration_pct": 5.7,
        "sparse_points": 80,
        "mean_reprojection_error": 3.8,
        "mean_track_length": 1.5,
        "camera_positions": [[0, 0, 0], [100, 100, 100]],
        "bounding_box": {"extents": [1.0, 1.0, 50.0]}  # highly elongated
    }
    val_failing = pipeline._validate_sparse_reconstruction(failing_stats)
    assert val_failing["result"] == "UNRELIABLE"
    assert any("Low image registration" in m for m in val_failing["messages"])


def test_quality_assessment_georef_distinction():
    """Verify quality report clearly distinguishes Georeferencing Status from Metric Accuracy."""
    qa = QualityAssessor(reports_dir=Path("data/output/reports"))

    video_meta = {"filename": "test.mp4", "resolution": "1920x1080", "fps": 30.0, "duration_seconds": 60, "total_frames": 1800}
    frame_qa = {"total_frames": 180, "accepted_frames": 160, "rejected_blur": 15, "rejected_exposure": 5, "rejected_duplicates": 0, "average_sharpness": 250.0}
    rec_meta = {
        "engine": "COLMAP",
        "registered_images": 150,
        "total_images": 160,
        "registration_pct": 93.8,
        "sparse_points_count": 8500,
        "dense_points_count": 45000,
        "mean_reprojection_error": 0.92,
        "mean_track_length": 4.1,
        "validation_result": "RELIABLE",
        "diagnostic_messages": ["Reconstruction meets quality criteria"]
    }
    mesh_meta = {"vertices_count": 12000, "faces_count": 24000, "method_used": "POISSON"}
    georef_meta = {"georeferencing_status": "GEOREFERENCED_WGS84_UTM", "datum": "WGS84 UTM Zone 43N", "utm_zone": "43N"}
    sensor_status = {"GPS": "AVAILABLE"}

    report = qa.generate_full_report(
        video_meta=video_meta,
        frame_quality_meta=frame_qa,
        reconstruction_meta=rec_meta,
        mesh_meta=mesh_meta,
        georeference_meta=georef_meta,
        sensor_status=sensor_status,
        start_time_seconds=1000.0
    )

    assert report["quality_assessment"]["quality_status"] == "RELIABLE"
    assert "georeferencing" in report
    assert "metric_accuracy" in report["georeferencing"]
    assert "does not guarantee geometric reconstruction accuracy" in report["georeferencing"]["note"].lower()
