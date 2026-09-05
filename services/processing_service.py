"""
Background Pipeline Processing Service — ElevateX SIH26158.

Orchestrates the 7 pipeline stages with thread-safe progress tracking.

Improvements:
  - Exposes full SfM metrics (registration %, reprojection error, sparse points).
  - Handles the sparse-validation halt gracefully (stops before dense MVS if SfM fails).
  - Reports diagnostic messages in real-time status.
  - mesh_generation and texture_generation are skipped when validation fails.
"""
import time
import threading
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

import config
from pipeline.video_processing import VideoProcessor
from pipeline.frame_extraction import FrameExtractor
from pipeline.frame_quality import FrameQualityAssessor
from pipeline.image_preprocessing import ImagePreprocessor
from pipeline.reconstruction import PhotogrammetryReconstructor
from pipeline.mesh_generation import MeshGenerator
from pipeline.texture_generation import TextureGenerator
from pipeline.georeferencing import Georeferencer
from pipeline.quality_assessment import QualityAssessor
from services.metadata_service import MetadataService


class ProcessingService:
    def __init__(self):
        self.lock = threading.Lock()
        self.status = {
            "state": "IDLE",  # IDLE, RUNNING, COMPLETED, FAILED
            "current_stage": None,
            "start_time": None,
            "elapsed_seconds": 0,
            "error": None,
            "stages": {
                "stage1_video":      {"name": "Video Ingestion & Extraction",   "status": "WAITING", "details": None},
                "stage2_quality":    {"name": "Frame Quality Analysis",          "status": "WAITING", "details": None},
                "stage3_sfm":        {"name": "SfM & Dense Reconstruction",      "status": "WAITING", "details": None},
                "stage4_mesh":       {"name": "Mesh Surface Generation",         "status": "WAITING", "details": None},
                "stage5_texture":    {"name": "Texture Baking & 3D Export",      "status": "WAITING", "details": None},
                "stage6_georef":     {"name": "Georeferencing & Spatial Alignment", "status": "WAITING", "details": None},
                "stage7_assessment": {"name": "Quality & Metric Assessment",     "status": "WAITING", "details": None},
            },
            "sensor_availability": {
                "GPS": "NOT AVAILABLE",
                "Flight Metadata": "NOT AVAILABLE",
                "IMU": "NOT AVAILABLE",
                "RTK": "NOT AVAILABLE",
                "Camera Intrinsics": "NOT AVAILABLE",
            },
            # Live reconstruction metrics (updated during stage 3)
            "reconstruction_metrics": {
                "total_input_frames": 0,
                "selected_keyframes": 0,
                "registered_images": 0,
                "registration_pct": 0.0,
                "sparse_points": 0,
                "dense_points": 0,
                "mean_reprojection_error": 0.0,
                "validation_result": "PENDING",
                "diagnostic_messages": [],
            },
            "artifacts": {
                "sparse_points_ply": None,
                "dense_points_ply": None,
                "mesh_ply": None,
                "model_glb": None,
                "quality_report": None,
            }
        }

    def get_status(self) -> Dict[str, Any]:
        with self.lock:
            if self.status["state"] == "RUNNING" and self.status["start_time"]:
                self.status["elapsed_seconds"] = round(
                    time.time() - self.status["start_time"], 1)
            return dict(self.status)

    def start_pipeline_async(self, video_path: str,
                             telemetry_path: Optional[str] = None):
        """Start the complete reconstruction pipeline on a background daemon thread."""
        thread = threading.Thread(
            target=self._run_pipeline,
            args=(video_path, telemetry_path),
            daemon=True
        )
        thread.start()

    # ------------------------------------------------------------------
    # Private pipeline runner
    # ------------------------------------------------------------------
    def _run_pipeline(self, video_path: str,
                      telemetry_path: Optional[str] = None):
        with self.lock:
            self.status["state"] = "RUNNING"
            self.status["start_time"] = time.time()
            self.status["error"] = None
            for k in self.status["stages"]:
                self.status["stages"][k]["status"] = "WAITING"
                self.status["stages"][k]["details"] = None
            # Reset metrics
            for k in self.status["reconstruction_metrics"]:
                if isinstance(self.status["reconstruction_metrics"][k], list):
                    self.status["reconstruction_metrics"][k] = []
                elif isinstance(self.status["reconstruction_metrics"][k], str):
                    self.status["reconstruction_metrics"][k] = "PENDING"
                else:
                    self.status["reconstruction_metrics"][k] = 0

        start_t = time.time()
        video_meta: Dict[str, Any] = {}
        frame_quality_meta: Dict[str, Any] = {}
        reconstruction_meta: Dict[str, Any] = {}
        mesh_meta: Dict[str, Any] = {}
        texture_meta: Dict[str, Any] = {}
        georef_meta: Dict[str, Any] = {}
        telemetry_data = []

        try:
            # ----------------------------------------------------------
            # Telemetry Pre-Check
            # ----------------------------------------------------------
            if telemetry_path and Path(telemetry_path).exists():
                parsed_tel = MetadataService.parse_flight_telemetry(telemetry_path)
                telemetry_data = parsed_tel.get("summary", {}).get("telemetry_data", [])
                with self.lock:
                    self.status["sensor_availability"] = \
                        MetadataService.get_sensor_status_summary(parsed_tel)

            # ----------------------------------------------------------
            # Stage 1: Video Ingestion & Frame Extraction
            # ----------------------------------------------------------
            self._set_stage("stage1_video", "RUNNING")

            v_proc = VideoProcessor(video_path)
            v_res = v_proc.validate_and_inspect()
            if not v_res["valid"]:
                raise ValueError(v_res.get("error", "Video validation failed"))
            video_meta = v_res["metadata"]

            f_ext = FrameExtractor()
            extract_res = f_ext.extract_frames(video_path)

            with self.lock:
                self.status["reconstruction_metrics"]["total_input_frames"] = \
                    extract_res.get("extracted_frames_count", 0)

            self._set_stage("stage1_video", "COMPLETED", {
                "resolution": video_meta["resolution"],
                "fps": video_meta["fps"],
                "extracted_frames": extract_res["extracted_frames_count"],
                "useful_video_frames": extract_res.get("useful_video_frames", 0),
                "outro_start_frame": extract_res.get("outro_start_frame", 0),
                "pre_rejected_blur": extract_res.get("pre_rejected_blur", 0),
                "pre_rejected_dark": extract_res.get("pre_rejected_dark", 0),
            })

            # ----------------------------------------------------------
            # Stage 2: Quality Filtering & Preprocessing
            # ----------------------------------------------------------
            self._set_stage("stage2_quality", "RUNNING")

            f_qa = FrameQualityAssessor()
            frame_quality_meta = f_qa.assess_and_filter()

            f_prep = ImagePreprocessor()
            prep_res = f_prep.preprocess_images()

            with self.lock:
                self.status["reconstruction_metrics"]["selected_keyframes"] = \
                    frame_quality_meta.get("accepted_frames", 0)

            self._set_stage("stage2_quality", "COMPLETED", {
                "accepted_frames": frame_quality_meta["accepted_frames"],
                "rejected_blur": frame_quality_meta["rejected_blur"],
                "rejected_duplicates": frame_quality_meta["rejected_duplicates"],
                "avg_sharpness": frame_quality_meta["average_sharpness"],
                "preprocessed": prep_res.get("preprocessed_count", 0),
                "sky_mask_applied": prep_res.get("sky_mask_applied", False),
            })

            # ----------------------------------------------------------
            # Stage 3: SfM & Dense Reconstruction (with Validation Gate)
            # ----------------------------------------------------------
            self._set_stage("stage3_sfm", "RUNNING")

            reconstructor = PhotogrammetryReconstructor()
            reconstruction_meta = reconstructor.run_reconstruction()

            validation = reconstruction_meta.get("validation_result", "UNKNOWN")
            diagnostics = reconstruction_meta.get("diagnostic_messages", [])

            with self.lock:
                m = self.status["reconstruction_metrics"]
                m["registered_images"] = reconstruction_meta.get("registered_images", 0)
                m["registration_pct"] = reconstruction_meta.get("registration_pct", 0.0)
                m["sparse_points"] = reconstruction_meta.get("sparse_points_count", 0)
                m["dense_points"] = reconstruction_meta.get("dense_points_count", 0)
                m["mean_reprojection_error"] = reconstruction_meta.get("mean_reprojection_error", 0.0)
                m["validation_result"] = validation
                m["diagnostic_messages"] = diagnostics
                self.status["artifacts"]["sparse_points_ply"] = \
                    reconstruction_meta.get("sparse_ply")
                self.status["artifacts"]["dense_points_ply"] = \
                    reconstruction_meta.get("clean_dense_ply") or \
                    reconstruction_meta.get("dense_ply")

            sfm_stage_status = "COMPLETED" if validation in ("RELIABLE", "MODERATE") else "FAILED"
            self._set_stage("stage3_sfm", sfm_stage_status, {
                "engine": reconstruction_meta.get("engine"),
                "sparse_points": reconstruction_meta.get("sparse_points_count", 0),
                "dense_points": reconstruction_meta.get("dense_points_count", 0),
                "registered_images": reconstruction_meta.get("registered_images", 0),
                "registration_pct": reconstruction_meta.get("registration_pct", 0.0),
                "mean_reprojection_error": reconstruction_meta.get("mean_reprojection_error", 0.0),
                "validation_result": validation,
                "diagnostic_messages": diagnostics,
            })

            # ----------------------------------------------------------
            # Stage 4: Mesh Surface Generation
            # (only if reconstruction is valid enough)
            # ----------------------------------------------------------
            has_point_cloud = bool(
                reconstruction_meta.get("clean_dense_ply") or
                reconstruction_meta.get("dense_ply") or
                reconstruction_meta.get("sparse_ply")
            )
            if validation in ("RELIABLE", "MODERATE") and has_point_cloud:
                self._set_stage("stage4_mesh", "RUNNING")
                try:
                    m_gen = MeshGenerator()
                    mesh_meta = m_gen.generate_mesh(
                        point_cloud_path=reconstruction_meta.get("clean_dense_ply") or
                                         reconstruction_meta.get("dense_ply") or
                                         reconstruction_meta.get("sparse_ply")
                    )
                    self._set_stage("stage4_mesh", "COMPLETED", {
                        "vertices": mesh_meta.get("vertices_count", 0),
                        "faces": mesh_meta.get("faces_count", 0),
                        "method": mesh_meta.get("method_used"),
                    })
                    with self.lock:
                        self.status["artifacts"]["mesh_ply"] = mesh_meta.get("mesh_ply")
                except Exception as mesh_err:
                    self._set_stage("stage4_mesh", "FAILED",
                                    {"error": str(mesh_err)})
                    mesh_meta = {}
            else:
                self._set_stage("stage4_mesh", "SKIPPED", {
                    "reason": f"Skipped — SfM validation result: {validation}. "
                              "Fix sparse reconstruction before mesh generation."
                })

            # ----------------------------------------------------------
            # Stage 5: Texture Generation & Web GLB Export
            # ----------------------------------------------------------
            mesh_ply = mesh_meta.get("mesh_ply") if mesh_meta else None
            if mesh_ply and Path(mesh_ply).exists():
                self._set_stage("stage5_texture", "RUNNING")
                try:
                    t_gen = TextureGenerator()
                    texture_meta = t_gen.generate_textured_model(mesh_path=mesh_ply)
                    self._set_stage("stage5_texture", "COMPLETED", {
                        "status": texture_meta.get("status"),
                        "formats": texture_meta.get("export_formats", []),
                    })
                    with self.lock:
                        self.status["artifacts"]["model_glb"] = texture_meta.get("model_glb")
                except Exception as tex_err:
                    self._set_stage("stage5_texture", "FAILED", {"error": str(tex_err)})
                    texture_meta = {}
            else:
                self._set_stage("stage5_texture", "SKIPPED", {
                    "reason": "No mesh available for texture generation."
                })

            # ----------------------------------------------------------
            # Stage 6: Georeferencing
            # ----------------------------------------------------------
            self._set_stage("stage6_georef", "RUNNING")
            georeferencer = Georeferencer()
            georef_meta = georeferencer.process_georeferencing(
                telemetry_data=telemetry_data)
            self._set_stage("stage6_georef", "COMPLETED", {
                "status": georef_meta.get("georeferencing_status"),
                "datum": georef_meta.get("datum", "Local Coordinate Space"),
            })

            # ----------------------------------------------------------
            # Stage 7: Quality Assessment Report
            # ----------------------------------------------------------
            self._set_stage("stage7_assessment", "RUNNING")

            qa = QualityAssessor()
            with self.lock:
                sensor_s = dict(self.status["sensor_availability"])

            quality_report = qa.generate_full_report(
                video_meta=video_meta,
                frame_quality_meta=frame_quality_meta,
                reconstruction_meta=reconstruction_meta,
                mesh_meta=mesh_meta,
                georeference_meta=georef_meta,
                sensor_status=sensor_s,
                start_time_seconds=start_t,
            )

            self._set_stage("stage7_assessment", "COMPLETED", {
                "quality_status": quality_report["quality_assessment"]["quality_status"],
                "registration_pct": quality_report["reconstruction_metrics"]["registration_percentage"],
                "sparse_points": quality_report["reconstruction_metrics"]["sparse_points"],
                "processing_time": quality_report["processing_time_seconds"],
            })

            with self.lock:
                self.status["artifacts"]["quality_report"] = quality_report
                self.status["state"] = "COMPLETED"
                self.status["current_stage"] = None

        except Exception as e:
            err_msg = str(e)
            trace = traceback.format_exc()
            with self.lock:
                self.status["state"] = "FAILED"
                self.status["error"] = err_msg
                curr = self.status["current_stage"]
                if curr and curr in self.status["stages"]:
                    self.status["stages"][curr]["status"] = "FAILED"
                    self.status["stages"][curr]["details"] = {
                        "error": err_msg, "trace": trace}

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------
    def _set_stage(self, stage_key: str, status: str,
                   details: Optional[Dict] = None):
        with self.lock:
            self.status["current_stage"] = stage_key if status == "RUNNING" else None
            self.status["stages"][stage_key]["status"] = status
            if details is not None:
                self.status["stages"][stage_key]["details"] = details
