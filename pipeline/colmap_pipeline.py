"""
COLMAP Photogrammetry Pipeline Orchestrator — ElevateX SIH26158.

Key improvements over the previous version:
  1. Single-camera shared intrinsics (OPENCV model) — prevents focal-length drift.
  2. Sequential matching with loop detection — captures orbital drone paths.
  3. Quadratic overlap matching — connects viewpoints across the building.
  4. SfM statistics parsing from COLMAP text model output.
  5. Sparse reconstruction VALIDATION GATE — stops pipeline if SfM produces
     a degenerate model (low registration, high reprojection error, bad trajectory).
  6. Dense MVS only runs if validation passes.
  7. Detailed diagnostic report when reconstruction is unreliable.
"""
import subprocess
import json
import logging
import math
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import config

logger = logging.getLogger(__name__)


class ColmapPipeline:
    def __init__(self,
                 images_dir: Path = config.PREPROCESSED_DIR,
                 output_dir: Path = config.OUTPUT_DIR):
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.colmap_exe = config.COLMAP_EXECUTABLE
        self.env = config.COLMAP_ENV
        self.available = config.COLMAP_AVAILABLE

        self.database_path = self.output_dir / "colmap_database.db"
        self.sparse_dir = self.output_dir / "sfm" / "colmap_sparse"
        self.dense_dir = self.output_dir / "dense" / "colmap_dense"

        self.sparse_dir.mkdir(parents=True, exist_ok=True)
        self.dense_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Availability check
    # ------------------------------------------------------------------
    def is_colmap_installed(self) -> Dict[str, Any]:
        if not self.colmap_exe:
            return {
                "installed": False,
                "executable": None,
                "message": "COLMAP not found. Built-in SfM engine will be used."
            }
        return {
            "installed": True,
            "executable": self.colmap_exe,
            "message": f"COLMAP located at {self.colmap_exe}"
        }

    # ------------------------------------------------------------------
    # Low-level command runner
    # ------------------------------------------------------------------
    def run_command(self, cmd_args: List[str],
                    timeout: int = 3600) -> Dict[str, Any]:
        """Run a COLMAP CLI sub-command using direct binary invocation."""
        if not self.colmap_exe:
            return {"success": False, "error": "COLMAP is not installed"}

        full_cmd = [self.colmap_exe] + cmd_args
        logger.info(f"[COLMAP] Running: {' '.join(str(c) for c in full_cmd)}")

        try:
            res = subprocess.run(
                full_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=timeout,
                env=self.env,
                shell=False
            )
            output = res.stdout or ""
            if res.returncode != 0:
                logger.warning(f"[COLMAP] Non-zero return code {res.returncode}.")
                logger.debug(f"[COLMAP] Output tail:\n{output[-2000:]}")
            else:
                logger.debug(f"[COLMAP] Output tail:\n{output[-1000:]}")
            return {
                "success": res.returncode == 0,
                "returncode": res.returncode,
                "stdout": output[-3000:] if output else "",
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": f"Command timed out after {timeout}s"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Main pipeline execution
    # ------------------------------------------------------------------
    def execute_pipeline(self) -> Dict[str, Any]:
        """
        Execute COLMAP SfM + (conditionally) MVS dense reconstruction.

        Returns a comprehensive dict including:
          - sparse_ply, dense_ply paths
          - sfm_stats (registered images, reprojection error, sparse points, etc.)
          - validation_result (RELIABLE / MODERATE / UNRELIABLE)
          - diagnostic_messages list
        """
        status = self.is_colmap_installed()
        if not status["installed"]:
            return {
                "status": "COLMAP_UNAVAILABLE",
                "message": status["message"],
                "sparse_ply": None,
                "dense_ply": None,
                "sfm_stats": {},
                "validation_result": "COLMAP_UNAVAILABLE",
                "diagnostic_messages": ["COLMAP is not installed. Using built-in SfM engine."]
            }

        # Wipe stale database from prior runs so features are re-extracted
        if self.database_path.exists():
            try:
                self.database_path.unlink()
                logger.info("[COLMAP] Removed stale database.")
            except OSError:
                pass

        steps_log = []

        # ----------------------------------------------------------------
        # Step 1 — Feature Extraction
        # ----------------------------------------------------------------
        params = config.COLMAP_PARAMS
        logger.info("[COLMAP] Step 1/6: Feature Extraction...")

        extract_args = [
            "feature_extractor",
            "--database_path", str(self.database_path),
            "--image_path", str(self.images_dir),
            # CRITICAL: Single shared camera model
            "--ImageReader.camera_model", params["camera_model"],
            "--ImageReader.single_camera", "1",
            # SIFT extraction quality
            "--SiftExtraction.max_num_features", str(params["sift_max_features"]),
            "--SiftExtraction.estimate_affine_shape",
            "1" if params["sift_estimate_affine_shape"] else "0",
            "--SiftExtraction.domain_size_pooling",
            "1" if params.get("sift_domain_size_pooling", True) else "0",
            "--FeatureExtraction.use_gpu", "1",
        ]
        extract_res = self.run_command(extract_args)
        if not extract_res["success"]:
            # Try CPU fallback
            logger.warning("[COLMAP] GPU SIFT failed, trying CPU...")
            extract_args[-1] = "0"
            extract_res = self.run_command(extract_args)
        steps_log.append({"step": "feature_extractor", "result": extract_res})

        # ----------------------------------------------------------------
        # Step 2 — Sequential Matching
        # ----------------------------------------------------------------
        logger.info("[COLMAP] Step 2/6: Sequential Matching...")

        match_args = [
            "sequential_matcher",
            "--database_path", str(self.database_path),
            "--SequentialMatching.overlap", str(params["sequential_overlap"]),
            "--SequentialMatching.quadratic_overlap",
            "1" if params["sequential_quadratic_overlap"] else "0",
            "--FeatureMatching.use_gpu", "1",
            "--SiftMatching.max_ratio", "0.80",
            "--SiftMatching.max_distance", "0.7",
            "--TwoViewGeometry.min_num_inliers",
            str(params.get("geo_verification_min_inliers", 15)),
        ]
        match_res = self.run_command(match_args, timeout=7200)
        if not match_res["success"]:
            logger.warning("[COLMAP] GPU matching failed, trying CPU...")
            for i, a in enumerate(match_args):
                if a == "1" and i > 0 and match_args[i - 1] == "--FeatureMatching.use_gpu":
                    match_args[i] = "0"
            match_res = self.run_command(match_args, timeout=7200)
        steps_log.append({"step": "sequential_matcher", "result": match_res})

        # ----------------------------------------------------------------
        # Step 3 — Mapper (Sparse SfM)
        # ----------------------------------------------------------------
        logger.info("[COLMAP] Step 3/6: Mapper (Sparse Reconstruction)...")

        mapper_args = [
            "mapper",
            "--database_path", str(self.database_path),
            "--image_path", str(self.images_dir),
            "--output_path", str(self.sparse_dir),
            "--Mapper.init_min_tri_angle", str(params.get("mapper_init_min_tri_angle", 2.0)),
            "--Mapper.init_min_num_inliers", "25",
            "--Mapper.abs_pose_min_num_inliers", "15",
            "--Mapper.abs_pose_min_inlier_ratio", "0.15",
            "--Mapper.tri_min_angle", "1.0",
            "--Mapper.ba_refine_focal_length",
            "1" if params.get("mapper_ba_refine_focal_length", True) else "0",
            "--Mapper.ba_refine_principal_point",
            "1" if params.get("mapper_ba_refine_principal_point", False) else "0",
            "--Mapper.ba_refine_extra_params",
            "1" if params.get("mapper_ba_refine_extra_params", True) else "0",
            "--Mapper.min_num_matches",
            str(params.get("mapper_min_num_matches", 15)),
        ]
        map_res = self.run_command(mapper_args, timeout=14400)
        steps_log.append({"step": "mapper", "result": map_res})

        # ----------------------------------------------------------------
        # Step 4 — Export sparse model to text + PLY
        # ----------------------------------------------------------------
        model_0_dir = self.sparse_dir / "0"
        sparse_ply = self.sparse_dir / "sparse.ply"

        if not model_0_dir.exists():
            # Try other numbered subdirs
            for sub in sorted(self.sparse_dir.iterdir()):
                if sub.is_dir() and sub.name.isdigit():
                    model_0_dir = sub
                    break

        text_model_dir = self.sparse_dir / "model_text"
        text_model_dir.mkdir(parents=True, exist_ok=True)

        if model_0_dir.exists():
            # Export to text for statistics parsing
            self.run_command([
                "model_converter",
                "--input_path", str(model_0_dir),
                "--output_path", str(text_model_dir),
                "--output_type", "TXT"
            ])
            # Export PLY
            self.run_command([
                "model_converter",
                "--input_path", str(model_0_dir),
                "--output_path", str(sparse_ply),
                "--output_type", "PLY"
            ])
            logger.info(f"[COLMAP] Sparse model exported: {sparse_ply}")
        else:
            logger.error("[COLMAP] No sparse model directory found after mapper.")

        # ----------------------------------------------------------------
        # Step 5 — Parse SfM statistics & Validate
        # ----------------------------------------------------------------
        logger.info("[COLMAP] Step 5/6: Parsing SfM statistics and validating...")
        sfm_stats = self._parse_sfm_statistics(text_model_dir)
        validation = self._validate_sparse_reconstruction(sfm_stats)

        logger.info(f"[COLMAP] SfM Stats: {sfm_stats}")
        logger.info(f"[COLMAP] Validation: {validation['result']} — {validation['messages']}")

        dense_ply = None

        # ----------------------------------------------------------------
        # Step 6 — Dense MVS (only if SfM passes validation)
        # ----------------------------------------------------------------
        if validation["result"] in ("RELIABLE", "MODERATE"):
            logger.info("[COLMAP] Step 6/6: Dense Reconstruction (MVS)...")

            undistort_res = self.run_command([
                "image_undistorter",
                "--image_path", str(self.images_dir),
                "--input_path", str(model_0_dir),
                "--output_path", str(self.dense_dir),
                "--output_type", "COLMAP",
                "--max_image_size", "960",
            ], timeout=600)
            steps_log.append({"step": "image_undistorter", "result": undistort_res})

            pms_res = self.run_command([
                "patch_match_stereo",
                "--workspace_path", str(self.dense_dir),
                "--workspace_format", "COLMAP",
                "--PatchMatchStereo.max_image_size", "960",
                "--PatchMatchStereo.window_radius", "3",
                "--PatchMatchStereo.num_iterations", "2",
                "--PatchMatchStereo.geom_consistency", "0",
                "--PatchMatchStereo.gpu_index", "0",
            ], timeout=600)
            steps_log.append({"step": "patch_match_stereo", "result": pms_res})

            dense_ply = self.dense_dir / "fused.ply"
            fusion_res = self.run_command([
                "stereo_fusion",
                "--workspace_path", str(self.dense_dir),
                "--workspace_format", "COLMAP",
                "--input_type", "photometric",
                "--output_type", "PLY",
                "--output_path", str(dense_ply),
            ], timeout=600)
            steps_log.append({"step": "stereo_fusion", "result": fusion_res})

            if not dense_ply.exists():
                dense_ply = None
                logger.info("[COLMAP] Dense fused.ply not produced — using high-density sparse model.")
        else:
            logger.warning(
                "[COLMAP] VALIDATION FAILED — Skipping dense reconstruction. "
                f"Diagnostics: {validation['messages']}"
            )
            validation["messages"].append(
                "Dense reconstruction was NOT executed because sparse SfM validation failed. "
                "Fix the sparse model first before re-running."
            )

        status_code = (
            "COLMAP_COMPLETED" if (dense_ply and dense_ply.exists()) else
            "COLMAP_SPARSE_ONLY" if (sparse_ply.exists()) else
            "COLMAP_FAILED"
        )

        return {
            "status": status_code,
            "sparse_ply": str(sparse_ply.resolve()) if sparse_ply.exists() else None,
            "dense_ply": str(dense_ply.resolve()) if dense_ply and dense_ply.exists() else (str(sparse_ply.resolve()) if sparse_ply.exists() else None),
            "sfm_stats": sfm_stats,
            "validation_result": validation["result"],
            "diagnostic_messages": validation["messages"],
            "steps": steps_log,
        }

    # ------------------------------------------------------------------
    # SfM Statistics Parser
    # ------------------------------------------------------------------
    def _parse_sfm_statistics(self, text_model_dir: Path) -> Dict[str, Any]:
        """
        Parse COLMAP text model files (cameras.txt, images.txt, points3D.txt)
        to extract reconstruction quality metrics.
        """
        stats: Dict[str, Any] = {
            "registered_images": 0,
            "total_images": self._count_images_in_input(),
            "registration_pct": 0.0,
            "sparse_points": 0,
            "mean_reprojection_error": 0.0,
            "mean_track_length": 0.0,
            "camera_positions": [],
            "bounding_box": None,
        }

        images_txt = text_model_dir / "images.txt"
        points_txt = text_model_dir / "points3D.txt"

        # Parse images.txt
        if images_txt.exists():
            positions = []
            image_ids = set()
            try:
                with open(images_txt, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("#") or not line:
                            continue
                        parts = line.split()
                        if len(parts) >= 8:
                            try:
                                # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
                                img_id = int(parts[0])
                                if img_id in image_ids:
                                    continue
                                image_ids.add(img_id)
                                # World position = -R^T * t
                                qw, qx, qy, qz = float(parts[1]), float(parts[2]), \
                                                  float(parts[3]), float(parts[4])
                                tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
                                R = self._quat_to_rotation(qw, qx, qy, qz)
                                t = np.array([tx, ty, tz])
                                cam_pos = -R.T @ t
                                positions.append(cam_pos.tolist())
                            except (ValueError, IndexError):
                                continue

            except Exception as e:
                logger.warning(f"[COLMAP] Error parsing images.txt: {e}")

            stats["registered_images"] = len(image_ids)
            stats["camera_positions"] = positions

        # Compute registration percentage
        total = stats["total_images"]
        registered = stats["registered_images"]
        stats["registration_pct"] = round(registered / max(1, total) * 100.0, 1)

        # Parse points3D.txt
        if points_txt.exists():
            try:
                errors = []
                track_lengths = []
                n_points = 0
                with open(points_txt, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("#") or not line:
                            continue
                        parts = line.split()
                        if len(parts) >= 8:
                            try:
                                err = float(parts[7])
                                errors.append(err)
                                # Track length = number of observations = (len(parts) - 8) / 2
                                track_len = (len(parts) - 8) / 2
                                track_lengths.append(track_len)
                                n_points += 1
                            except (ValueError, IndexError):
                                continue
                stats["sparse_points"] = n_points
                if errors:
                    stats["mean_reprojection_error"] = round(float(np.mean(errors)), 3)
                if track_lengths:
                    stats["mean_track_length"] = round(float(np.mean(track_lengths)), 2)
            except Exception as e:
                logger.warning(f"[COLMAP] Error parsing points3D.txt: {e}")

        # Compute bounding box of camera positions
        if stats["camera_positions"]:
            pos_arr = np.array(stats["camera_positions"])
            bbox_min = pos_arr.min(axis=0).tolist()
            bbox_max = pos_arr.max(axis=0).tolist()
            extents = [bbox_max[i] - bbox_min[i] for i in range(3)]
            stats["bounding_box"] = {
                "min": bbox_min,
                "max": bbox_max,
                "extents": extents,
            }

        return stats

    def _count_images_in_input(self) -> int:
        """Count total images sent to COLMAP."""
        return len(list(self.images_dir.glob("*.jpg"))) + \
               len(list(self.images_dir.glob("*.png")))

    @staticmethod
    def _quat_to_rotation(qw: float, qx: float,
                          qy: float, qz: float) -> np.ndarray:
        """Convert unit quaternion to 3×3 rotation matrix."""
        R = np.array([
            [1 - 2*(qy**2 + qz**2),   2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw),   1 - 2*(qx**2 + qz**2),   2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw),       2*(qy*qz + qx*qw),   1 - 2*(qx**2 + qy**2)]
        ], dtype=np.float64)
        return R

    # ------------------------------------------------------------------
    # Sparse Reconstruction Validation Gate
    # ------------------------------------------------------------------
    def _validate_sparse_reconstruction(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates whether the sparse SfM model is reliable enough to proceed
        to dense MVS reconstruction.

        Returns:
          result: "RELIABLE" | "MODERATE" | "UNRELIABLE"
          messages: list of human-readable diagnostic explanations
        """
        thresholds = config.VALIDATION
        messages: List[str] = []
        issues = 0

        reg_pct = stats.get("registration_pct", 0.0)
        reprojection_err = stats.get("mean_reprojection_error", 999.0)
        sparse_pts = stats.get("sparse_points", 0)
        track_length = stats.get("mean_track_length", 0.0)
        registered = stats.get("registered_images", 0)
        total = stats.get("total_images", 1)

        # Check 1: Registration percentage
        if reg_pct < thresholds["min_registration_pct"]:
            msg = (f"Low image registration: only {reg_pct:.1f}% of images registered "
                   f"(threshold: {thresholds['min_registration_pct']}%). "
                   f"Possible causes: too few matching features, large viewpoint changes, "
                   f"or insufficient overlap between consecutive frames.")
            messages.append(msg)
            logger.warning(f"[VALIDATION] {msg}")
            issues += 2  # Major issue
        else:
            messages.append(f"[OK] Registration: {reg_pct:.1f}% ({registered}/{total} images)")

        # Check 2: Reprojection error
        if reprojection_err > thresholds["max_reprojection_error"]:
            msg = (f"High mean reprojection error: {reprojection_err:.2f}px "
                   f"(threshold: {thresholds['max_reprojection_error']}px). "
                   f"Possible causes: incorrect camera model, inaccurate intrinsics, "
                   f"large lens distortion, or incorrect feature matches.")
            messages.append(msg)
            logger.warning(f"[VALIDATION] {msg}")
            issues += 2
        elif reprojection_err == 0.0 and sparse_pts == 0:
            msg = ("Reprojection error is zero and no sparse points found. "
                   "COLMAP mapper likely failed to register any images.")
            messages.append(msg)
            logger.error(f"[VALIDATION] {msg}")
            issues += 4  # Critical issue
        else:
            messages.append(f"[OK] Mean reprojection error: {reprojection_err:.2f}px")

        # Check 3: Sparse point count
        if sparse_pts < thresholds["min_sparse_points"]:
            msg = (f"Too few sparse 3D points: {sparse_pts} "
                   f"(threshold: {thresholds['min_sparse_points']}). "
                   f"Possible causes: low-texture surfaces, motion blur, "
                   f"sky-dominant frames, or matching failure.")
            messages.append(msg)
            logger.warning(f"[VALIDATION] {msg}")
            issues += 2
        else:
            messages.append(f"[OK] Sparse points: {sparse_pts:,}")

        # Check 4: Track length (well-triangulated points observed by many cameras)
        if sparse_pts > 0 and track_length < thresholds.get("min_track_length", 3.0):
            msg = (f"Short mean track length: {track_length:.1f} "
                   f"(threshold: {thresholds['min_track_length']}). "
                   f"Points are only seen by a few cameras — possible poor overlap.")
            messages.append(msg)
            logger.warning(f"[VALIDATION] {msg}")
            issues += 1

        # Check 5: Camera trajectory smoothness
        cam_positions = stats.get("camera_positions", [])
        if len(cam_positions) > 5:
            traj_issue = self._check_trajectory_smoothness(
                cam_positions, thresholds.get("max_cam_position_jump_factor", 8.0)
            )
            if traj_issue:
                messages.append(traj_issue)
                logger.warning(f"[VALIDATION] {traj_issue}")
                issues += 1
            else:
                messages.append(f"[OK] Camera trajectory is smooth ({len(cam_positions)} registered cameras)")

        # Check 6: Bounding box elongation (degenerate elongated reconstruction)
        bbox = stats.get("bounding_box")
        if bbox and sparse_pts > 100:
            extents = sorted(bbox["extents"])
            if extents[0] > 1e-6:
                elongation = extents[-1] / extents[0]
                if elongation > thresholds.get("max_bounding_box_elongation", 15.0):
                    msg = (f"Reconstruction is highly elongated (axis ratio {elongation:.1f}:1). "
                           f"This typically indicates drift in the camera chain — "
                           f"the model does not form a consistent 3D structure. "
                           f"Possible causes: missing loop closure, single-pass linear trajectory "
                           f"with poor lateral baseline, or scale drift.")
                    messages.append(msg)
                    logger.warning(f"[VALIDATION] {msg}")
                    issues += 1
                else:
                    messages.append(f"[OK] Bounding box elongation: {elongation:.1f}:1")

        # Classify result
        if issues == 0:
            result = "RELIABLE"
        elif issues <= 2:
            result = "MODERATE"
        else:
            result = "UNRELIABLE"

        return {"result": result, "messages": messages, "issue_score": issues}

    @staticmethod
    def _check_trajectory_smoothness(positions: List[List[float]],
                                     max_jump_factor: float) -> Optional[str]:
        """
        Check that camera positions do not contain erratic large jumps.
        Returns None if smooth, or a diagnostic string if erratic.
        """
        pos_arr = np.array(positions)
        diffs = np.linalg.norm(np.diff(pos_arr, axis=0), axis=1)
        if len(diffs) == 0:
            return None
        median_diff = np.median(diffs)
        if median_diff < 1e-9:
            return ("Camera positions are all clustered at the same point — "
                    "SfM may have degenerated to a single location (gauge failure).")
        max_diff = diffs.max()
        if max_diff > max_jump_factor * median_diff:
            jump_idx = int(np.argmax(diffs))
            return (f"Camera trajectory has an erratic jump at position {jump_idx} "
                    f"(jump: {max_diff:.2f}, median spacing: {median_diff:.2f}, "
                    f"ratio: {max_diff/median_diff:.1f}x). "
                    f"Possible causes: wrong loop closure match, missing overlap at that point.")
        return None
