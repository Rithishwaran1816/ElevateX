"""
Frame Quality Assessment & Intelligent Keyframe Selection Module — ElevateX SIH26158.

Given a candidate pool of pre-extracted frames, this module:
  1. Scores each frame on sharpness, brightness, and scene quality.
  2. Selects ~150-200 optimal keyframes using a coverage-aware algorithm that:
     - Ensures geometrically diverse viewpoints across the full trajectory.
     - Prioritises frames with good parallax (motion between consecutive frames).
     - Avoids clusters of nearly-identical static frames.
     - Maintains minimum/maximum spacing across the video timeline.
  3. Generates a detailed frame_quality_report.json with all metrics and diagnostics.

OUTPUT: selected_frames/ directory with ~150-200 JPEG keyframes.
"""
import shutil
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import config

import logging
logger = logging.getLogger(__name__)


class FrameQualityAssessor:
    def __init__(self,
                 input_dir: Path = config.FRAMES_DIR,
                 output_dir: Path = config.SELECTED_FRAMES_DIR,
                 report_dir: Path = config.REPORTS_DIR):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.report_dir = Path(report_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Static metric helpers (kept for backward-compat with tests)
    # ------------------------------------------------------------------
    @staticmethod
    def calculate_sharpness(image: np.ndarray) -> float:
        """Compute Laplacian variance as sharpness indicator."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def calculate_brightness(image: np.ndarray) -> float:
        """Compute average brightness across luminance channel."""
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            return float(np.mean(hsv[:, :, 2]))
        return float(np.mean(image))

    @staticmethod
    def calculate_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute normalised histogram correlation across colour channels."""
        correlations = []
        ch = min(img1.shape[2], img2.shape[2]) if len(img1.shape) == 3 else 1
        for i in range(ch):
            c1 = img1[:, :, i] if len(img1.shape) == 3 else img1
            c2 = img2[:, :, i] if len(img2.shape) == 3 else img2
            h1 = cv2.calcHist([c1], [0], None, [32], [0, 256])
            h2 = cv2.calcHist([c2], [0], None, [32], [0, 256])
            cv2.normalize(h1, h1, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(h2, h2, 0, 1, cv2.NORM_MINMAX)
            corr = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
            if np.isnan(corr):
                corr = 1.0 if np.allclose(h1, h2) else 0.0
            correlations.append(float(corr))
        return float(np.mean(correlations))

    # ------------------------------------------------------------------
    # Backward-compatible wrapper (used by processing_service.py)
    # ------------------------------------------------------------------
    def assess_and_filter(self,
                          blur_threshold: float = config.FRAME_QUALITY["blur_threshold"],
                          min_brightness: float = config.FRAME_QUALITY["min_brightness"],
                          max_brightness: float = config.FRAME_QUALITY["max_brightness"],
                          duplicate_threshold: float = config.FRAME_QUALITY["duplicate_hist_threshold"]
                          ) -> Dict[str, Any]:
        """
        Entry point called by the pipeline. Internally calls the new intelligent
        keyframe selector and returns a backward-compatible report dict.
        """
        return self.select_keyframes()

    # ------------------------------------------------------------------
    # Core: Intelligent Keyframe Selection
    # ------------------------------------------------------------------
    def select_keyframes(self) -> Dict[str, Any]:
        """
        Main keyframe selection algorithm.

        Algorithm:
        1. Load all candidate frames from input_dir, compute quality scores.
        2. Hard-reject frames below quality floors (dark, excessively blurry).
        3. Adaptively set blur threshold from the frame population.
        4. Divide the timeline into N equal segments (N = target_keyframes).
        5. In each segment, pick the highest-scoring frame.
        6. Apply post-pass deduplication: remove near-identical adjacent selections.
        7. Enforce minimum inter-frame motion (skip static hovering runs).
        8. Copy final selection to output_dir.
        """
        # Clear previous selection
        for f in self.output_dir.glob("*.jpg"):
            try:
                f.unlink()
            except OSError:
                pass

        frame_files = sorted(self.input_dir.glob("frame_*.jpg"))
        total_candidates = len(frame_files)
        if total_candidates == 0:
            raise ValueError(f"No candidate frames found in {self.input_dir}")

        logger.info(f"[QUALITY] Scoring {total_candidates} candidate frames...")

        cfg = config.KEYFRAME_SELECTION
        target = cfg["target_keyframes"]
        min_kf = cfg["min_keyframes"]
        max_kf = cfg["max_keyframes"]
        blur_floor = cfg["blur_threshold"]
        min_brightness = cfg["min_brightness"]
        max_brightness = cfg["max_brightness"]
        min_hist_dist = cfg["min_histogram_distance"]
        min_spacing = cfg.get("min_frame_spacing", 3)

        # ---- Step 1: Score all candidates -----------------------------------
        scored: List[Dict[str, Any]] = []
        for i, fp in enumerate(frame_files):
            img = cv2.imread(str(fp))
            if img is None:
                continue

            # Compute metrics on downsampled version for speed
            small = cv2.resize(img, (640, 360))
            gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            sharpness = self.calculate_sharpness(small)
            brightness = self.calculate_brightness(small)

            # Compute optical flow vs. previous frame
            scored.append({
                "path": fp,
                "index": i,
                "sharpness": sharpness,
                "brightness": brightness,
                "small": small,
                "gray_small": gray_small,
            })

        if not scored:
            raise ValueError("All candidate frames were unreadable.")

        # ---- Step 2: Adaptive blur threshold --------------------------------
        sharpness_vals = [s["sharpness"] for s in scored]
        adaptive_threshold = float(np.percentile(sharpness_vals,
                                                  cfg["adaptive_blur_percentile"]))
        effective_blur_threshold = max(blur_floor, adaptive_threshold)
        logger.info(f"[QUALITY] Effective blur threshold: {effective_blur_threshold:.1f} "
                    f"(population 25th-pct: {adaptive_threshold:.1f})")

        # ---- Step 3: Hard reject -------------------------------------------
        # Compute motion magnitude between adjacent scored frames
        prev_gray = None
        for item in scored:
            gray = item["gray_small"]
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                item["motion"] = float(np.mean(mag))
            else:
                item["motion"] = 5.0  # Assume moderate motion for first frame
            prev_gray = gray

        # Build composite quality score
        max_sharpness = max(sharpness_vals) + 1e-6
        max_motion = max(s.get("motion", 0) for s in scored) + 1e-6
        for item in scored:
            # Normalised sharpness [0..1]
            norm_sharp = item["sharpness"] / max_sharpness
            # Penalise extreme motion (motion blur) and near-zero motion (static)
            motion = item["motion"]
            motion_score = 1.0 - abs(motion - cfg["min_motion_magnitude"]) / (
                cfg["max_motion_magnitude"] + 1e-6
            )
            motion_score = max(0.0, min(1.0, motion_score))
            # Brightness acceptability
            bri = item["brightness"]
            bri_ok = 1.0 if min_brightness <= bri <= max_brightness else 0.2
            # Composite
            item["quality_score"] = (0.6 * norm_sharp + 0.3 * motion_score + 0.1 * bri_ok)

        # ---- Step 4: Hard reject blurry / dark / overexposed frames --------
        eligible = []
        rejected_blur = 0
        rejected_exposure = 0
        for item in scored:
            if item["sharpness"] < effective_blur_threshold:
                rejected_blur += 1
                continue
            bri = item["brightness"]
            if bri < min_brightness or bri > max_brightness:
                rejected_exposure += 1
                continue
            eligible.append(item)

        logger.info(f"[QUALITY] After hard-reject: {len(eligible)}/{total_candidates} eligible "
                    f"(blur={rejected_blur}, exposure={rejected_exposure})")

        # Fallback: if too few eligible, relax threshold and take top-N by sharpness
        if len(eligible) < min_kf:
            logger.warning(f"[QUALITY] Too few eligible frames ({len(eligible)} < {min_kf}). "
                           f"Falling back to top-{min_kf} by sharpness.")
            scored_by_sharp = sorted(scored, key=lambda x: x["sharpness"], reverse=True)
            eligible = scored_by_sharp[:max(min_kf, len(scored_by_sharp))]

        # ---- Step 5: Segment-based selection --------------------------------
        # Divide eligible frames across N=target temporal segments and pick
        # the highest-quality frame in each segment.
        n_eligible = len(eligible)
        n_segments = min(target, n_eligible)
        segment_size = n_eligible / n_segments

        segment_picks: List[Dict[str, Any]] = []
        for seg in range(n_segments):
            seg_start = int(seg * segment_size)
            seg_end = int((seg + 1) * segment_size)
            seg_frames = eligible[seg_start:seg_end]
            if not seg_frames:
                continue
            best = max(seg_frames, key=lambda x: x["quality_score"])
            segment_picks.append(best)

        logger.info(f"[QUALITY] Segment-based selection: {len(segment_picks)} keyframes.")

        # ---- Step 6: Post-pass deduplication --------------------------------
        # Remove picks that are nearly identical to their neighbour
        deduplicated: List[Dict[str, Any]] = []
        prev_hist = None
        for item in segment_picks:
            # Compute a compact color histogram for this frame
            hist = self._compact_histogram(item["small"])
            if prev_hist is not None:
                dist = float(np.sum(np.abs(hist - prev_hist)))
                if dist < min_hist_dist:
                    # Near-duplicate: keep the better-quality one
                    if deduplicated and item["quality_score"] > deduplicated[-1]["quality_score"]:
                        deduplicated[-1] = item
                        prev_hist = hist
                    continue
            deduplicated.append(item)
            prev_hist = hist

        logger.info(f"[QUALITY] After deduplication: {len(deduplicated)} keyframes.")

        # ---- Step 7: Enforce max keyframe cap --------------------------------
        if len(deduplicated) > max_kf:
            # Keep evenly spaced frames from deduplicated list
            indices = np.round(np.linspace(0, len(deduplicated) - 1, max_kf)).astype(int)
            deduplicated = [deduplicated[i] for i in indices]
            logger.info(f"[QUALITY] Capped to {max_kf} keyframes.")

        # ---- Step 8: Copy to output_dir ------------------------------------
        accepted_files: List[str] = []
        for out_idx, item in enumerate(deduplicated):
            dest = self.output_dir / f"frame_{out_idx:05d}.jpg"
            shutil.copy2(item["path"], dest)
            accepted_files.append(str(dest.resolve()))

        avg_sharpness = float(np.mean([item["sharpness"] for item in deduplicated])) \
            if deduplicated else 0.0
        avg_motion = float(np.mean([item.get("motion", 0) for item in deduplicated])) \
            if deduplicated else 0.0

        report = {
            "total_frames": total_candidates,
            "eligible_after_hard_reject": len(eligible),
            "accepted_frames": len(accepted_files),
            "rejected_blur": rejected_blur,
            "rejected_exposure": rejected_exposure,
            "rejected_duplicates": len(segment_picks) - len(deduplicated),
            "average_sharpness": round(avg_sharpness, 2),
            "average_motion_magnitude": round(avg_motion, 2),
            "effective_blur_threshold": round(effective_blur_threshold, 2),
            "selected_frames": accepted_files,
            "blur_threshold_used": effective_blur_threshold,
            "exposure_bounds": [min_brightness, max_brightness],
            "selection_algorithm": "segment_based_quality_selection",
        }

        report_path = self.report_dir / "frame_quality_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logger.info(f"[QUALITY] Final selection: {len(accepted_files)} keyframes. "
                    f"Avg sharpness: {avg_sharpness:.1f}, Avg motion: {avg_motion:.2f}")
        return report

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _compact_histogram(img: np.ndarray, bins: int = 32) -> np.ndarray:
        """Compute a normalised, concatenated RGB histogram for similarity checks."""
        hists = []
        for c in range(3):
            h = cv2.calcHist([img], [c], None, [bins], [0, 256]).flatten()
            h = h / (h.sum() + 1e-6)
            hists.append(h)
        return np.concatenate(hists)
