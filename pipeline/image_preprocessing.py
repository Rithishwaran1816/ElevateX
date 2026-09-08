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

    @staticmethod
    def apply_dynamic_object_mask(image: np.ndarray, yolo_model=None, conf_threshold: float = 0.35) -> np.ndarray:
        """
        Detect and mask dynamic objects (people, vehicles, moving objects) to prevent
        false SIFT correspondences across drone video frames.
        Masks the detected bounding box by applying smooth Gaussian blur / inpainting.
        """
        # Dynamic object class IDs in COCO (person=0, bicycle=1, car=2, motorcycle=3,
        # airplane=4, bus=5, train=6, truck=7, boat=8, bird=14, cat=15, dog=16, horse=17, sheep=18, cow=19)
        DYNAMIC_CLASSES = {0, 1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19}
        if yolo_model is None:
            return image

        try:
            results = yolo_model(image, verbose=False, conf=conf_threshold)
            out_img = image.copy()
            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    cls_id = int(box.cls[0])
                    if cls_id in DYNAMIC_CLASSES:
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = xyxy
                        h, w = image.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        if x2 > x1 and y2 > y1:
                            roi = out_img[y1:y2, x1:x2]
                            # Blur / smooth the dynamic object region to eliminate high-frequency SIFT corners
                            blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)
                            out_img[y1:y2, x1:x2] = blurred_roi
            return out_img
        except Exception as e:
            logger.warning(f"[PREPROC] Dynamic object masking error: {e}")
            return image

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

        yolo_model = None
        if mask_dynamic_objects:
            try:
                from ultralytics import YOLO
                yolo_model = YOLO("yolov8n.pt")
                logger.info("[PREPROC] Loaded YOLO model for dynamic object masking.")
            except Exception as e:
                logger.warning(f"[PREPROC] Could not initialize YOLO for dynamic masking: {e}")

        for f in image_files:
            img = cv2.imread(str(f))
            if img is None:
                logger.warning(f"[PREPROC] Skipping unreadable file: {f}")
                continue

            processed = img.copy()

            # 1. Dynamic object masking (vehicles, people)
            if mask_dynamic_objects and yolo_model is not None:
                processed = self.apply_dynamic_object_mask(processed, yolo_model=yolo_model)

            # 2. CLAHE illumination normalisation
            if enable_clahe:
                processed = self.apply_clahe(
                    processed,
                    clip_limit=config.IMAGE_PREPROCESSING["clahe_clip_limit"],
                    grid_size=config.IMAGE_PREPROCESSING["clahe_grid_size"]
                )

            # 3. Edge-preserving denoising
            if enable_denoising:
                processed = self.apply_subtle_denoising(processed)

            # 4. Soft sky suppression
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
            "dynamic_masking_applied": mask_dynamic_objects and (yolo_model is not None),
            "files": processed_files
        }
