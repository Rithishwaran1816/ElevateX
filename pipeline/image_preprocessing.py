"""
Image Preprocessing & Illumination Normalisation Module — ElevateX SIH26158.

Applies:
  1. CLAHE (Contrast Limited Adaptive Histogram Equalisation) for shadow/highlight
     recovery on building facades and chimneys.
  2. Edge-preserving bilateral denoising — preserves high-frequency textures
     needed by SIFT feature extraction.
  3. Optional sky-region soft suppression: reduces high-frequency noise and
     uniform sky gradients that generate false SIFT features.

Does NOT aggressively crop frames — static environmental context is preserved
to help COLMAP feature matching and camera registration.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import config
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    def __init__(self,
                 input_dir: Path = config.SELECTED_FRAMES_DIR,
                 output_dir: Path = config.PREPROCESSED_DIR):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Static processing helpers
    # ------------------------------------------------------------------
    @staticmethod
    def apply_clahe(image: np.ndarray,
                    clip_limit: float = 2.0,
                    grid_size: tuple = (8, 8)) -> np.ndarray:
        """Apply CLAHE on Luminance channel (LAB colour space)."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        l_clahe = clahe.apply(l)
        lab_enhanced = cv2.merge((l_clahe, a, b))
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    @staticmethod
    def apply_subtle_denoising(image: np.ndarray) -> np.ndarray:
        """Bilateral filter — preserves edges needed for SIFT without smoothing them."""
        return cv2.bilateralFilter(image, d=5, sigmaColor=35, sigmaSpace=35)

    @staticmethod
    def apply_sky_mask(image: np.ndarray,
                       sky_upper_fraction: float = 0.25,
                       strength: float = 0.5) -> np.ndarray:
        """
        Soft-suppress sky region to reduce false feature matches on uniform sky.

        Uses a gradient alpha mask so there is no sharp hard boundary.
        Preserves building roof / chimney features visible at the top of the frame.

        strength=0: no effect. strength=1: fully darkens sky to suppress features.
        """
        h, w = image.shape[:2]
        sky_height = int(h * sky_upper_fraction)
        if sky_height <= 0:
            return image

        result = image.copy().astype(np.float32)

        # Build a gradient alpha mask: 1.0 at top, 0.0 at sky_height boundary
        alpha = np.linspace(1.0, 0.0, sky_height, dtype=np.float32)
        alpha_map = alpha[:, np.newaxis, np.newaxis]  # (sky_height, 1, 1)

        # Blend toward a darkened version in the sky zone
        sky_region = result[:sky_height]
        darkened = sky_region * (1.0 - strength * alpha_map)
        result[:sky_height] = darkened

        return np.clip(result, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Main preprocessing entry point
    # ------------------------------------------------------------------
    def preprocess_images(self,
                          enable_clahe: bool = config.IMAGE_PREPROCESSING["enable_clahe"],
                          enable_denoising: bool = config.IMAGE_PREPROCESSING["enable_denoising"],
                          mask_dynamic_objects: bool = config.IMAGE_PREPROCESSING["enable_dynamic_object_masking"],
                          enable_sky_mask: bool = config.IMAGE_PREPROCESSING.get("enable_sky_mask", True)
                          ) -> Dict[str, Any]:
        """
        Process all selected keyframes and output preprocessed images.
        """
        # Clear output directory
        for f in self.output_dir.glob("*.jpg"):
            try:
                f.unlink()
            except OSError:
                pass

        image_files = sorted(self.input_dir.glob("*.jpg"))
        if not image_files:
            raise ValueError(f"No selected frames to preprocess in {self.input_dir}")

        processed_files: List[str] = []
        sky_fraction = config.IMAGE_PREPROCESSING.get("sky_upper_fraction",
                       config.KEYFRAME_SELECTION.get("sky_upper_fraction", 0.25))
        sky_strength = config.IMAGE_PREPROCESSING.get("sky_mask_strength", 0.5)

        for f in image_files:
            img = cv2.imread(str(f))
            if img is None:
                logger.warning(f"[PREPROC] Skipping unreadable file: {f}")
                continue

            processed = img.copy()

            # 1. CLAHE illumination normalisation
            if enable_clahe:
                processed = self.apply_clahe(
                    processed,
                    clip_limit=config.IMAGE_PREPROCESSING["clahe_clip_limit"],
                    grid_size=config.IMAGE_PREPROCESSING["clahe_grid_size"]
                )

            # 2. Edge-preserving denoising
            if enable_denoising:
                processed = self.apply_subtle_denoising(processed)

            # 3. Soft sky suppression
            if enable_sky_mask:
                processed = self.apply_sky_mask(processed,
                                                sky_upper_fraction=sky_fraction,
                                                strength=sky_strength)

            out_file = self.output_dir / f.name
            cv2.imwrite(str(out_file), processed, [cv2.IMWRITE_JPEG_QUALITY, 98])
            processed_files.append(str(out_file.resolve()))

        logger.info(f"[PREPROC] Preprocessed {len(processed_files)} images → {self.output_dir}")
        return {
            "preprocessed_count": len(processed_files),
            "output_directory": str(self.output_dir.resolve()),
            "clahe_applied": enable_clahe,
            "denoising_applied": enable_denoising,
            "sky_mask_applied": enable_sky_mask,
            "dynamic_masking_applied": mask_dynamic_objects,
            "files": processed_files
        }
