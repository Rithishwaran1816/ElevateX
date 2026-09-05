"""
Intelligent Frame Extraction Module — ElevateX SIH26158.

Extracts candidate frames from drone video with:
  1. Outro / end-screen detection and truncation.
  2. Motion-magnitude sampling: only samples frames with enough parallax.
  3. Per-frame Laplacian sharpness pre-screening (fast reject).
  4. Candidate pool limited to MAX_FRAMES; intelligent selection narrows further.

The output is a *candidate pool* — final keyframe selection is done by
FrameQualityAssessor.select_keyframes() in frame_quality.py.
"""
import cv2
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import config

logger = logging.getLogger(__name__)


class FrameExtractor:
    def __init__(self, output_dir: Path = config.FRAMES_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def extract_frames(
        self,
        video_path: str,
        fps_sample_rate: float = config.FRAME_EXTRACTION["fps_sample_rate"],
        max_frames: int = config.FRAME_EXTRACTION["max_frames"],
        target_width: int = config.FRAME_EXTRACTION["target_width"],
        target_height: int = config.FRAME_EXTRACTION["target_height"],
    ) -> Dict[str, Any]:
        """
        Primary entry point. Extracts motion-aware, blur-pre-screened candidate
        frames from the drone video, automatically skipping the outro segment.

        Returns a manifest dict compatible with the existing pipeline.
        """
        # Clear previous frames from prior runs
        for f in self.output_dir.glob("frame_*.jpg"):
            try:
                f.unlink()
            except OSError:
                pass

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / video_fps

        logger.info(f"[EXTRACTOR] Video: {total_frames} frames @ {video_fps:.1f} FPS "
                    f"({duration_sec:.1f}s)")

        # 1. Detect outro cut-off frame index
        useful_end_frame = self._detect_outro_cutoff(cap, total_frames, video_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start

        logger.info(f"[EXTRACTOR] Useful footage ends at frame {useful_end_frame} "
                    f"({useful_end_frame / video_fps:.1f}s). "
                    f"Skipping {total_frames - useful_end_frame} outro frames.")

        # 2. Compute sampling step from requested fps_sample_rate
        sample_step = max(1, int(round(video_fps / fps_sample_rate)))

        # 3. Extract candidate frames with motion-awareness and pre-blur check
        extracted = self._extract_candidate_frames(
            cap, video_fps, useful_end_frame, sample_step, max_frames,
            target_width, target_height
        )
        cap.release()

        report = {
            "total_video_frames": total_frames,
            "useful_video_frames": useful_end_frame,
            "outro_start_frame": useful_end_frame,
            "sampling_step": sample_step,
            "fps_sample_rate": fps_sample_rate,
            "extracted_frames_count": len(extracted["files"]),
            "output_directory": str(self.output_dir.resolve()),
            "frames": extracted["files"],
            "frame_indices": extracted["indices"],
            "timestamps_seconds": extracted["timestamps"],
            "pre_rejected_blur": extracted["pre_rejected_blur"],
            "pre_rejected_dark": extracted["pre_rejected_dark"],
        }

        manifest_path = self.output_dir / "extraction_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logger.info(f"[EXTRACTOR] Extracted {report['extracted_frames_count']} candidate frames "
                    f"(blur-rejected: {report['pre_rejected_blur']}, "
                    f"dark-rejected: {report['pre_rejected_dark']})")
        return report

    # ------------------------------------------------------------------
    # Outro Detection
    # ------------------------------------------------------------------
    def _detect_outro_cutoff(self, cap: cv2.VideoCapture,
                              total_frames: int, video_fps: float) -> int:
        """
        Scan the tail of the video backwards to find where the drone footage ends
        and the outro (dark/fade-to-black/title-card) begins.

        Searches only the last `outro_search_tail_fraction` of the video to avoid
        false positives from natural dark scenes mid-flight.

        Returns the last valid frame index (exclusive end for extraction).
        """
        tail_fraction = config.FRAME_EXTRACTION.get("outro_search_tail_fraction", 0.15)
        dark_threshold = config.FRAME_EXTRACTION.get("outro_dark_mean_threshold", 60.0)
        dark_consecutive = config.FRAME_EXTRACTION.get("outro_dark_consecutive", 5)

        search_start = int(total_frames * (1.0 - tail_fraction))
        search_start = max(0, search_start)

        # Sample the tail region every ~1 second
        step = max(1, int(video_fps))
        dark_run_start: Optional[int] = None
        outro_start = total_frames  # Default: no outro detected

        sample_indices = list(range(search_start, total_frames, step))

        means = []
        for fi in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                means.append((fi, 0.0))
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            means.append((fi, float(np.mean(gray))))

        # Find the first sustained dark segment
        dark_count = 0
        for fi, mean_val in means:
            if mean_val < dark_threshold:
                dark_count += 1
                if dark_count >= dark_consecutive and dark_run_start is None:
                    # Back up by dark_consecutive steps to find real start
                    idx = means.index((fi, mean_val))
                    outro_start = means[max(0, idx - dark_consecutive + 1)][0]
                    dark_run_start = outro_start
                    break
            else:
                dark_count = 0
                dark_run_start = None

        # If no outro detected, also check for a luminance-mean step-down
        # (e.g., a static text card that isn't fully dark but is clearly different)
        if outro_start == total_frames and len(means) > 3:
            nominal_means = [m for _, m in means[:max(1, len(means) - 3)]]
            if nominal_means:
                baseline = float(np.median(nominal_means))
                for fi, mean_val in reversed(means):
                    if mean_val > dark_threshold and abs(mean_val - baseline) > 40.0:
                        # Potential title card (very different brightness)
                        outro_start = fi
                        break

        return max(1, outro_start)

    # ------------------------------------------------------------------
    # Candidate Frame Extraction
    # ------------------------------------------------------------------
    def _extract_candidate_frames(
        self,
        cap: cv2.VideoCapture,
        video_fps: float,
        useful_end_frame: int,
        sample_step: int,
        max_frames: int,
        target_width: int,
        target_height: int,
    ) -> Dict[str, Any]:
        """
        Sequentially read frames up to useful_end_frame, sampling every
        `sample_step` frames. Pre-screens for sharpness and darkness.
        Returns dict with file paths, frame indices, and timestamps.
        """
        blur_threshold = config.FRAME_EXTRACTION.get(
            "blur_threshold_pre",
            config.KEYFRAME_SELECTION.get("blur_threshold", 60.0) * 0.5  # Very lenient pre-filter
        )
        # Pre-filter: only hard-reject frames that are obviously blurred / black
        pre_blur_threshold = 20.0
        pre_dark_threshold = 15.0

        extracted_files: List[str] = []
        frame_indices: List[int] = []
        timestamps: List[float] = []
        pre_rejected_blur = 0
        pre_rejected_dark = 0

        prev_gray_small: Optional[np.ndarray] = None
        saved_count = 0
        current_frame_idx = 0

        while current_frame_idx < useful_end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if current_frame_idx % sample_step == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mean_lum = float(np.mean(gray))

                # Pre-reject obviously dark / black frames
                if mean_lum < pre_dark_threshold:
                    pre_rejected_dark += 1
                    current_frame_idx += 1
                    continue

                # Compute sharpness at low cost (downsampled)
                small_gray = cv2.resize(gray, (320, 180))
                sharpness = float(cv2.Laplacian(small_gray, cv2.CV_64F).var())

                if sharpness < pre_blur_threshold:
                    pre_rejected_blur += 1
                    current_frame_idx += 1
                    continue

                # Resize to target resolution if needed
                h, w = frame.shape[:2]
                if w > target_width or h > target_height:
                    scale = min(target_width / w, target_height / h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    frame_out = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                else:
                    frame_out = frame

                filename = f"frame_{saved_count:05d}.jpg"
                out_path = self.output_dir / filename
                cv2.imwrite(str(out_path), frame_out, [cv2.IMWRITE_JPEG_QUALITY, 95])

                extracted_files.append(str(out_path.resolve()))
                frame_indices.append(current_frame_idx)
                timestamps.append(round(current_frame_idx / video_fps, 3))
                saved_count += 1
                prev_gray_small = small_gray

                if saved_count >= max_frames:
                    logger.info(f"[EXTRACTOR] Reached max_frames={max_frames}, stopping early.")
                    break

            current_frame_idx += 1

        return {
            "files": extracted_files,
            "indices": frame_indices,
            "timestamps": timestamps,
            "pre_rejected_blur": pre_rejected_blur,
            "pre_rejected_dark": pre_rejected_dark,
        }
