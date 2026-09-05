"""
Comprehensive Quality Assessment Module — ElevateX SIH26158.

Synthesises all pipeline metrics and generates the official quality_report.json.

Improvements:
  - Full reconstruction quality metrics: registration %, reprojection error,
    sparse points, dense points, track length.
  - Three-tier quality classification: RELIABLE / MODERATE / UNRELIABLE.
  - Diagnostic explanations for each quality tier.
  - Clear separation of Georeferencing Status vs Metric Accuracy vs Reconstruction Quality.
  - Processing duration tracking.
"""
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import config


class QualityAssessor:
    def __init__(self, reports_dir: Path = config.REPORTS_DIR):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_full_report(self,
                             video_meta: Dict[str, Any],
                             frame_quality_meta: Dict[str, Any],
                             reconstruction_meta: Dict[str, Any],
                             mesh_meta: Dict[str, Any],
                             georeference_meta: Dict[str, Any],
                             sensor_status: Dict[str, str],
                             start_time_seconds: float) -> Dict[str, Any]:
        """
        Generate the full consolidated quality report.
        Does NOT fabricate metric accuracy — reflects actual photogrammetry output.
        """
        elapsed = round(time.time() - start_time_seconds, 2)

        # ----------------------------------------------------------------
        # Frame metrics
        # ----------------------------------------------------------------
        total_extracted = frame_quality_meta.get("total_frames", 0)
        accepted_frames = frame_quality_meta.get("accepted_frames", 0)
        avg_sharpness = frame_quality_meta.get("average_sharpness", 0.0)

        # ----------------------------------------------------------------
        # Reconstruction metrics
        # ----------------------------------------------------------------
        registered_images = reconstruction_meta.get("registered_images", 0)
        total_images = reconstruction_meta.get("total_images", accepted_frames)
        registration_pct = reconstruction_meta.get("registration_pct",
            round(registered_images / max(1, total_images) * 100.0, 1))

        sparse_pts = reconstruction_meta.get("sparse_points_count", 0)
        dense_pts = reconstruction_meta.get("dense_points_count", 0)
        mean_reproj_err = reconstruction_meta.get("mean_reprojection_error", 0.0)
        mean_track_length = reconstruction_meta.get("mean_track_length", 0.0)

        vertices = mesh_meta.get("vertices_count", 0)
        faces = mesh_meta.get("faces_count", 0)

        # ----------------------------------------------------------------
        # Quality status classification
        # ----------------------------------------------------------------
        validation_result = reconstruction_meta.get("validation_result", "UNKNOWN")
        diagnostic_messages = reconstruction_meta.get("diagnostic_messages", [])

        quality_status, quality_explanation = self._classify_quality(
            validation_result=validation_result,
            registration_pct=registration_pct,
            mean_reproj_err=mean_reproj_err,
            sparse_pts=sparse_pts,
            dense_pts=dense_pts,
            diagnostic_messages=diagnostic_messages
        )

        # ----------------------------------------------------------------
        # Coverage assessment
        # ----------------------------------------------------------------
        if dense_pts > 50000:
            coverage_grade = "HIGH_DENSITY"
        elif dense_pts > 5000:
            coverage_grade = "MEDIUM_DENSITY"
        elif dense_pts > 0:
            coverage_grade = "SPARSE_COVERAGE"
        else:
            coverage_grade = "NO_DENSE_RECONSTRUCTION"

        # ----------------------------------------------------------------
        # Georeferencing — clearly separated from metric accuracy
        # ----------------------------------------------------------------
        georef_status = georeference_meta.get("georeferencing_status", "LOCAL_COORDINATE_SYSTEM")
        has_georef = georef_status == "GEOREFERENCED_WGS84_UTM"

        # Metric accuracy is SEPARATE from georeferencing
        if has_georef and quality_status == "RELIABLE":
            metric_accuracy_note = (
                "Georeferenced (WGS84/UTM). Metric scale is referenced to GPS anchor. "
                "Absolute accuracy depends on GPS precision (~3–5m without RTK)."
            )
        elif has_georef:
            metric_accuracy_note = (
                "Georeferenced (WGS84/UTM), but 3D reconstruction quality is limited. "
                "GPS coordinates are available but geometric accuracy is not guaranteed."
            )
        else:
            metric_accuracy_note = (
                "No georeferencing. 3D model is in a local relative Euclidean space. "
                "Scale and absolute position are not metrically calibrated."
            )

        # ----------------------------------------------------------------
        # Build report
        # ----------------------------------------------------------------
        report = {
            "project_name": "ElevateX — Drone 3D Reconstruction System",
            "problem_statement": "SIH26158",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "processing_time_seconds": elapsed,
            "reconstruction_engine": reconstruction_meta.get("engine", "UNKNOWN"),

            # Frame & keyframe quality
            "frame_quality": {
                "total_input_frames": total_extracted,
                "selected_keyframes": accepted_frames,
                "rejected_blur": frame_quality_meta.get("rejected_blur", 0),
                "rejected_exposure": frame_quality_meta.get("rejected_exposure", 0),
                "rejected_duplicates": frame_quality_meta.get("rejected_duplicates", 0),
                "average_sharpness": avg_sharpness,
                "blur_threshold_used": frame_quality_meta.get("blur_threshold_used", 0),
                "selection_algorithm": frame_quality_meta.get("selection_algorithm", "N/A"),
            },

            # Video properties
            "video_properties": {
                "filename": video_meta.get("filename"),
                "resolution": video_meta.get("resolution"),
                "fps": video_meta.get("fps"),
                "duration_seconds": video_meta.get("duration_seconds"),
                "total_video_frames": video_meta.get("total_frames"),
            },

            # Core reconstruction metrics
            "reconstruction_metrics": {
                "registered_images": registered_images,
                "total_input_images": total_images,
                "registration_percentage": registration_pct,
                "sparse_points": sparse_pts,
                "dense_points": dense_pts,
                "mesh_vertices": vertices,
                "mesh_faces": faces,
                "mean_reprojection_error_pixels": mean_reproj_err,
                "mean_track_length": mean_track_length,
                "surface_coverage_assessment": coverage_grade,
            },

            # Quality judgement
            "quality_assessment": {
                "quality_status": quality_status,          # RELIABLE / MODERATE / UNRELIABLE
                "quality_explanation": quality_explanation,
                "diagnostic_messages": diagnostic_messages,
                "validation_result": validation_result,
            },

            # Sensor integration status
            "sensor_availability": sensor_status,

            # Georeferencing — EXPLICITLY separated from metric accuracy
            "georeferencing": {
                "georeferencing_status": georef_status,
                "coordinate_system": georeference_meta.get("datum", "Local Euclidean"),
                "utm_zone": georeference_meta.get("utm_zone"),
                "bounding_box": georeference_meta.get("spatial_bounding_box"),
                # NOTE: Georeferencing ≠ Geometric Accuracy
                "note": (
                    "GPS/WGS84 coordinates indicate spatial reference only. "
                    "Georeferencing does NOT guarantee geometric reconstruction accuracy. "
                    "See 'metric_accuracy' for reconstruction quality."
                ),
                "metric_accuracy": metric_accuracy_note,
            },

            # Future modules
            "future_modules_readiness": {
                "neural_radiance_fields_nerf": "ARCHITECTED",
                "3d_gaussian_splatting": "ARCHITECTED",
                "ai_depth_super_resolution": "ARCHITECTED"
            }
        }

        report_file = self.reports_dir / "quality_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report

    # ------------------------------------------------------------------
    # Quality classification
    # ------------------------------------------------------------------
    @staticmethod
    def _classify_quality(
        validation_result: str,
        registration_pct: float,
        mean_reproj_err: float,
        sparse_pts: int,
        dense_pts: int,
        diagnostic_messages: List[str]
    ):
        """Determine quality tier and produce a human-readable explanation."""

        if validation_result == "RELIABLE":
            status = "RELIABLE"
            explanation = (
                f"Reconstruction is reliable. "
                f"{registration_pct:.1f}% of images registered, "
                f"mean reprojection error {mean_reproj_err:.2f}px, "
                f"{sparse_pts:,} sparse and {dense_pts:,} dense points. "
                "The 3D model should geometrically represent the photographed structure."
            )
        elif validation_result == "MODERATE":
            status = "MODERATE"
            explanation = (
                f"Reconstruction quality is moderate. "
                f"{registration_pct:.1f}% of images registered, "
                f"mean reprojection error {mean_reproj_err:.2f}px. "
                "Some regions may be incomplete or inaccurate. "
                "Review diagnostic messages for specific issues."
            )
        elif validation_result in ("UNRELIABLE", "FAILED"):
            status = "UNRELIABLE"
            explanation = (
                f"Reconstruction quality is unreliable. "
                f"Only {registration_pct:.1f}% of images registered "
                f"with {sparse_pts:,} sparse points. "
                "Dense reconstruction may have been skipped. "
                "The 3D model does NOT reliably represent the actual scene. "
                "See diagnostic messages for root causes."
            )
        elif validation_result == "COLMAP_UNAVAILABLE":
            status = "UNRELIABLE"
            explanation = (
                "COLMAP is not available. "
                "Built-in SfM engine was used as a fallback. "
                "Results are approximate and may not accurately represent the scene."
            )
        else:
            # Unknown / built-in fallback
            if registration_pct >= 70 and sparse_pts > 1000:
                status = "MODERATE"
                explanation = (
                    f"Built-in SfM registered {registration_pct:.1f}% of images. "
                    f"Results are approximate."
                )
            else:
                status = "UNRELIABLE"
                explanation = (
                    f"Reconstruction quality could not be verified. "
                    f"Registration: {registration_pct:.1f}%, Sparse points: {sparse_pts:,}."
                )

        return status, explanation
