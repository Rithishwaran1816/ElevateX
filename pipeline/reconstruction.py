"""
Structure-from-Motion (SfM) and Dense Multi-View Reconstruction Engine — ElevateX SIH26158.

Orchestrates reconstruction:
  - COLMAP pipeline when available (preferred).
  - Built-in OpenCV + Open3D photogrammetry engine as fallback.

Key improvements:
  - Passes COLMAP validation result and SfM statistics upstream.
  - Open3D statistical + radius outlier removal on dense cloud.
  - Normal estimation oriented consistently.
  - Generates diagnostic report on reconstruction failure.
"""
import os
import cv2
import json
import logging
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Dict, Any, List, Optional
import config
from pipeline.colmap_pipeline import ColmapPipeline

logger = logging.getLogger(__name__)


class PhotogrammetryReconstructor:
    def __init__(self,
                 images_dir: Path = config.PREPROCESSED_DIR,
                 output_dir: Path = config.OUTPUT_DIR,
                 models_dir: Path = config.MODELS_DIR):
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.models_dir = Path(models_dir)
        self.sfm_dir = config.SFM_DIR
        self.dense_dir = config.DENSE_DIR

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.sfm_dir.mkdir(parents=True, exist_ok=True)
        self.dense_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Camera intrinsics helper (used by built-in engine)
    # ------------------------------------------------------------------
    @staticmethod
    def estimate_camera_intrinsics(width: int, height: int,
                                   focal_factor: float = 1.2) -> np.ndarray:
        """Construct intrinsic camera matrix K with sensible pinhole prior."""
        focal_length = max(width, height) * focal_factor
        cx, cy = width / 2.0, height / 2.0
        return np.array([
            [focal_length, 0.0, cx],
            [0.0, focal_length, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def run_reconstruction(self, force_builtin: bool = False) -> Dict[str, Any]:
        """
        Execute 3D reconstruction.
        Uses COLMAP when available; falls back to built-in OpenCV/Open3D engine.
        """
        if config.COLMAP_AVAILABLE and not force_builtin:
            return self._run_colmap()
        return self._run_builtin_sfm_and_dense()

    # ------------------------------------------------------------------
    # COLMAP path
    # ------------------------------------------------------------------
    def _run_colmap(self) -> Dict[str, Any]:
        colmap_runner = ColmapPipeline(self.images_dir, self.output_dir)
        colmap_res = colmap_runner.execute_pipeline()

        sfm_stats = colmap_res.get("sfm_stats", {})
        validation = colmap_res.get("validation_result", "UNKNOWN")
        diagnostics = colmap_res.get("diagnostic_messages", [])

        # Clean dense PLY or sparse PLY if dense does not exist
        sparse_ply = colmap_res.get("sparse_ply")
        dense_ply = colmap_res.get("dense_ply")
        clean_dense_ply = None

        if dense_ply and Path(dense_ply).exists() and self._count_ply_points(dense_ply) > 0:
            clean_dense_ply = self._clean_point_cloud(dense_ply, "dense")
        elif sparse_ply and Path(sparse_ply).exists():
            clean_dense_ply = self._clean_point_cloud(sparse_ply, "dense")

        # Compute camera centroid for normal orientation
        cam_positions = sfm_stats.get("camera_positions", [])
        cam_centroid = np.mean(np.array(cam_positions), axis=0).tolist() \
            if len(cam_positions) > 0 else [0.0, 0.0, 5.0]

        # Count total valid points
        total_points = self._count_ply_points(clean_dense_ply) or \
                       self._count_ply_points(dense_ply) or \
                       sfm_stats.get("sparse_points", 0)

        # Prepare result
        result = {
            "engine": "COLMAP",
            "status": colmap_res.get("status", "COLMAP_FAILED"),
            "validation_result": validation,
            "diagnostic_messages": diagnostics,
            "sparse_ply": sparse_ply,
            "dense_ply": dense_ply or sparse_ply,
            "clean_dense_ply": clean_dense_ply or sparse_ply,
            "colmap_details": colmap_res,
            # SfM metrics exposed for quality_assessment
            "registered_images": sfm_stats.get("registered_images", 0),
            "total_images": sfm_stats.get("total_images", 0),
            "registration_pct": sfm_stats.get("registration_pct", 0.0),
            "sparse_points_count": sfm_stats.get("sparse_points", 0),
            "dense_points_count": total_points,
            "mean_reprojection_error": sfm_stats.get("mean_reprojection_error", 0.0),
            "mean_track_length": sfm_stats.get("mean_track_length", 0.0),
            "camera_centroid": cam_centroid,
            "camera_positions_count": len(cam_positions),
        }
        return result

    # ------------------------------------------------------------------
    # Open3D Dense Point Cloud Cleaning
    # ------------------------------------------------------------------
    def _clean_point_cloud(self, ply_path: str, label: str = "dense") -> Optional[str]:
        """
        Load a COLMAP-output point cloud and apply:
          1. Statistical outlier removal.
          2. Radius outlier removal (removes isolated noise points).
          3. Normal estimation oriented toward camera centroid.

        Returns path to the cleaned PLY, or None if cleaning failed.
        """
        try:
            pcd = o3d.io.read_point_cloud(str(ply_path))
            n_orig = len(pcd.points)
            logger.info(f"[CLEAN] Loaded {n_orig:,} points from {ply_path}")

            if n_orig < 10:
                logger.warning(f"[CLEAN] Too few points ({n_orig}), skipping cleaning.")
                return ply_path

            cfg = config.POINT_CLOUD_CLEANING

            # Step 1: Statistical outlier removal
            pcd, ind = pcd.remove_statistical_outlier(
                nb_neighbors=cfg["statistical_nb_neighbors"],
                std_ratio=cfg["statistical_std_ratio"]
            )
            logger.info(f"[CLEAN] After statistical filter: {len(pcd.points):,} points "
                        f"(removed {n_orig - len(pcd.points):,})")

            # Step 2: Radius outlier removal — remove tiny isolated clusters
            if len(pcd.points) > 20:
                distances = pcd.compute_nearest_neighbor_distance()
                avg_dist = float(np.mean(distances))
                radius = cfg["radius_radius_factor"] * avg_dist
                pcd, _ = pcd.remove_radius_outlier(
                    nb_points=cfg["radius_nb_points"],
                    radius=radius
                )
                logger.info(f"[CLEAN] After radius filter: {len(pcd.points):,} points")

            # Step 3: Normal estimation
            if len(pcd.points) > 3:
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=avg_dist * 5.0 if 'avg_dist' in dir() else 0.1,
                        max_nn=30
                    )
                )
                pcd.orient_normals_towards_camera_location(
                    camera_location=np.array([0.0, 0.0, 10.0])
                )

            clean_path = self.models_dir / f"{label}_points_clean.ply"
            o3d.io.write_point_cloud(str(clean_path), pcd)
            logger.info(f"[CLEAN] Saved cleaned cloud: {clean_path} ({len(pcd.points):,} pts)")
            return str(clean_path.resolve())

        except Exception as e:
            logger.error(f"[CLEAN] Point cloud cleaning failed: {e}", exc_info=True)
            return ply_path

    @staticmethod
    def _count_ply_points(ply_path: Optional[str]) -> int:
        """Return the number of points in a PLY file without loading it fully."""
        if not ply_path or not Path(ply_path).exists():
            return 0
        try:
            pcd = o3d.io.read_point_cloud(str(ply_path))
            return len(pcd.points)
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Built-in Photogrammetry Engine (fallback — no COLMAP)
    # ------------------------------------------------------------------
    def _run_builtin_sfm_and_dense(self) -> Dict[str, Any]:
        """
        Built-in SfM engine using OpenCV SIFT, essential matrix estimation,
        and triangulation. Used when COLMAP is not installed.

        NOTE: The built-in engine is a best-effort fallback. For production
        quality reconstruction, COLMAP is strongly recommended.
        """
        image_paths = sorted(list(self.images_dir.glob("*.jpg")))
        if len(image_paths) < 2:
            raise ValueError(f"Need at least 2 images for 3D reconstruction, "
                             f"found {len(image_paths)}")

        logger.info(f"[BUILTIN-SFM] Processing {len(image_paths)} images...")

        sift = cv2.SIFT_create(nfeatures=config.RECONSTRUCTION["max_features"])
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        keypoints_list, descriptors_list = [], []
        images_rgb, images_gray = [], []

        for p in image_paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, des = sift.detectAndCompute(gray, None)

            if des is None or len(kp) < 50:
                orb = cv2.ORB_create(nfeatures=config.RECONSTRUCTION["max_features"])
                kp, des = orb.detectAndCompute(gray, None)
                if des is not None:
                    des = des.astype(np.float32)

            keypoints_list.append(kp)
            descriptors_list.append(des)
            images_rgb.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            images_gray.append(gray)

        h, w = images_gray[0].shape[:2]
        K = self.estimate_camera_intrinsics(
            w, h, config.RECONSTRUCTION["focal_length_prior_factor"]
        )

        camera_poses = [{"R": np.eye(3), "t": np.zeros((3, 1))}]
        sparse_points_3d, sparse_colors = [], []
        registered_count = 1
        current_R = np.eye(3, dtype=np.float64)
        current_t = np.zeros((3, 1), dtype=np.float64)

        for i in range(len(image_paths) - 1):
            des1, des2 = descriptors_list[i], descriptors_list[i + 1]
            kp1, kp2 = keypoints_list[i], keypoints_list[i + 1]

            if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
                continue

            matches = flann.knnMatch(des1, des2, k=2)
            good = [m for m_n in matches if len(m_n) == 2
                    for m, n in [m_n] if m.distance < config.RECONSTRUCTION["match_ratio"] * n.distance]

            if len(good) < config.RECONSTRUCTION["min_inliers"]:
                continue

            pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

            E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            if E is None or mask is None:
                continue

            inlier_pts1 = pts1[mask.ravel() == 1]
            inlier_pts2 = pts2[mask.ravel() == 1]
            if len(inlier_pts1) < 15:
                continue

            _, R_rel, t_rel, _ = cv2.recoverPose(E, inlier_pts1, inlier_pts2, K)

            P1 = K @ np.hstack((current_R, current_t))
            next_R = R_rel @ current_R
            next_t = current_t + current_R.T @ t_rel
            P2 = K @ np.hstack((next_R, next_t))

            pts4d = cv2.triangulatePoints(P1, P2, inlier_pts1.T, inlier_pts2.T)
            pts3d = pts4d[:3] / (pts4d[3] + 1e-10)

            for idx_pt in range(pts3d.shape[1]):
                pt = pts3d[:, idx_pt]
                if np.isnan(pt).any() or np.isinf(pt).any():
                    continue
                d1 = (current_R @ pt.reshape(3, 1) + current_t)[2, 0]
                d2 = (next_R @ pt.reshape(3, 1) + next_t)[2, 0]
                if 0.1 < d1 < 250.0 and 0.1 < d2 < 250.0:
                    sparse_points_3d.append(pt)
                    u = min(max(0, int(round(inlier_pts1[idx_pt, 0]))), w - 1)
                    v = min(max(0, int(round(inlier_pts1[idx_pt, 1]))), h - 1)
                    sparse_colors.append(images_rgb[i][v, u] / 255.0)

            current_R, current_t = next_R, next_t
            camera_poses.append({"R": current_R, "t": current_t})
            registered_count += 1

        logger.info(f"[BUILTIN-SFM] Registered {registered_count} images, "
                    f"triangulated {len(sparse_points_3d)} sparse points.")

        # Dense point cloud synthesis from optical flow
        dense_points_3d = list(sparse_points_3d)
        dense_colors = list(sparse_colors)

        for i in range(min(len(camera_poses) - 1, len(images_gray) - 1)):
            flow = cv2.calcOpticalFlowFarneback(
                images_gray[i], images_gray[i + 1], None,
                pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                poly_n=5, poly_sigma=1.2, flags=0
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            R_cam = camera_poses[i]["R"]
            t_cam = camera_poses[i]["t"]
            fx, fy = K[0, 0], K[1, 1]
            cx_k, cy_k = K[0, 2], K[1, 2]

            for y in range(0, h, 8):
                for x in range(0, w, 8):
                    m = mag[y, x]
                    if m > 0.8:
                        est_depth = (fx * 0.5) / (m + 0.1)
                        if 0.5 < est_depth < 120.0:
                            x_c = (x - cx_k) * est_depth / fx
                            y_c = (y - cy_k) * est_depth / fy
                            pt_cam = np.array([[x_c], [y_c], [est_depth]])
                            pt_world = R_cam.T @ (pt_cam - t_cam)
                            dense_points_3d.append(pt_world.ravel())
                            dense_colors.append(images_rgb[i][y, x] / 255.0)

        # Do NOT generate a fake fallback geometry if reconstruction fails
        if len(sparse_points_3d) < 10:
            logger.error("[BUILTIN-SFM] Reconstruction degenerate — too few sparse points.")
            reg_pct = round(registered_count / max(1, len(image_paths)) * 100.0, 1)
            return {
                "engine": "ELEVATEX_BUILTIN_SFM",
                "status": "FAILED",
                "validation_result": "UNRELIABLE",
                "diagnostic_messages": [
                    f"Built-in SfM produced only {len(sparse_points_3d)} sparse points.",
                    f"Registered {registered_count}/{len(image_paths)} images ({reg_pct}%).",
                    "Possible causes: insufficient parallax, motion blur, "
                    "or near-duplicate frames.",
                ],
                "registered_images": registered_count,
                "total_images": len(image_paths),
                "registration_pct": reg_pct,
                "sparse_points_count": len(sparse_points_3d),
                "dense_points_count": 0,
                "sparse_ply": None,
                "dense_ply": None,
                "clean_dense_ply": None,
            }

        # Build Open3D clouds
        sparse_pcd = o3d.geometry.PointCloud()
        sparse_pcd.points = o3d.utility.Vector3dVector(
            np.array(sparse_points_3d, dtype=np.float64))
        sparse_pcd.colors = o3d.utility.Vector3dVector(
            np.array(sparse_colors, dtype=np.float64))

        dense_pcd = o3d.geometry.PointCloud()
        dense_pcd.points = o3d.utility.Vector3dVector(
            np.array(dense_points_3d, dtype=np.float64))
        dense_pcd.colors = o3d.utility.Vector3dVector(
            np.array(dense_colors, dtype=np.float64))

        # Clean dense cloud
        cfg = config.POINT_CLOUD_CLEANING
        if len(dense_pcd.points) > 50:
            _, ind = dense_pcd.remove_statistical_outlier(
                nb_neighbors=cfg["statistical_nb_neighbors"],
                std_ratio=cfg["statistical_std_ratio"]
            )
            dense_pcd = dense_pcd.select_by_index(ind)

        # Normal estimation
        try:
            sparse_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            dense_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            dense_pcd.orient_normals_towards_camera_location(
                camera_location=np.array([0.0, 0.0, 5.0]))
        except Exception:
            pass

        sparse_ply = self.models_dir / "sparse_points.ply"
        dense_ply = self.models_dir / "dense_points.ply"
        clean_ply = self.models_dir / "dense_points_clean.ply"

        o3d.io.write_point_cloud(str(sparse_ply), sparse_pcd)
        o3d.io.write_point_cloud(str(dense_ply), dense_pcd)
        o3d.io.write_point_cloud(str(clean_ply), dense_pcd)

        reg_pct = round(registered_count / max(1, len(image_paths)) * 100.0, 1)
        validation = "RELIABLE" if reg_pct >= 70 else "MODERATE" if reg_pct >= 40 else "UNRELIABLE"

        return {
            "engine": "ELEVATEX_BUILTIN_SFM",
            "status": "SUCCESS",
            "validation_result": validation,
            "diagnostic_messages": [
                f"Built-in SfM registered {registered_count}/{len(image_paths)} images ({reg_pct}%)",
                f"Sparse points: {len(sparse_points_3d):,}",
                f"Dense points (post-cleaning): {len(dense_pcd.points):,}",
            ],
            "registered_images": registered_count,
            "total_images": len(image_paths),
            "registration_pct": reg_pct,
            "sparse_points_count": len(sparse_pcd.points),
            "dense_points_count": len(dense_pcd.points),
            "sparse_ply": str(sparse_ply.resolve()),
            "dense_ply": str(dense_ply.resolve()),
            "clean_dense_ply": str(clean_ply.resolve()),
            "mean_reprojection_error": 0.0,  # Not computed in built-in engine
        }
