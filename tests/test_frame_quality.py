"""
Unit Tests for Frame Quality Assessment and Duplicate Filtering.
"""
import pytest
import numpy as np
import cv2
from pipeline.frame_quality import FrameQualityAssessor

def test_sharpness_calculation():
    # Sharp image: high contrast checkerboard
    sharp_img = np.zeros((200, 200), dtype=np.uint8)
    sharp_img[::20, :] = 255
    sharp_img[:, ::20] = 255

    # Blurry image: heavily Gaussian blurred version
    blur_img = cv2.GaussianBlur(sharp_img, (31, 31), 15)

    sharp_score = FrameQualityAssessor.calculate_sharpness(sharp_img)
    blur_score = FrameQualityAssessor.calculate_sharpness(blur_img)

    assert sharp_score > blur_score, "Sharp image must have higher Laplacian variance than blurred image"
    assert sharp_score > 500.0, "High contrast grid should have high sharpness"
    assert blur_score < sharp_score * 0.1, "Blurred image should have low sharpness relative to sharp image"

def test_brightness_calculation():
    dark_img = np.full((100, 100, 3), 15, dtype=np.uint8)
    bright_img = np.full((100, 100, 3), 240, dtype=np.uint8)

    dark_val = FrameQualityAssessor.calculate_brightness(dark_img)
    bright_val = FrameQualityAssessor.calculate_brightness(bright_img)

    assert dark_val < 30.0, "Dark image should report low brightness"
    assert bright_val > 200.0, "Bright image should report high brightness"

def test_duplicate_similarity():
    img1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    img2 = img1.copy() # Exact duplicate
    img3 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) # Distinct image

    sim_dup = FrameQualityAssessor.calculate_similarity(img1, img2)
    sim_diff = FrameQualityAssessor.calculate_similarity(img1, img3)

    assert abs(sim_dup - 1.0) < 0.02, "Identical frames must have histogram correlation near 1.0"
    assert sim_dup > sim_diff
