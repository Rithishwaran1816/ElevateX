"""
ElevateX System Configuration
Defines directory paths, algorithm parameters, and tool discovery.
"""
import os
import shutil
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
VIDEOS_DIR = INPUT_DIR / "videos"
METADATA_DIR = INPUT_DIR / "metadata"
GPS_DIR = INPUT_DIR / "gps"

OUTPUT_DIR = DATA_DIR / "output"
FRAMES_DIR = OUTPUT_DIR / "frames"
SELECTED_FRAMES_DIR = OUTPUT_DIR / "selected_frames"
PREPROCESSED_DIR = OUTPUT_DIR / "preprocessed_frames"
SFM_DIR = OUTPUT_DIR / "sfm"
DENSE_DIR = OUTPUT_DIR / "dense"
MODELS_DIR = OUTPUT_DIR / "models"
REPORTS_DIR = OUTPUT_DIR / "reports"
UPLOADS_DIR = BASE_DIR / "uploads"
LOGS_DIR = OUTPUT_DIR / "logs"

# Ensure all essential directories exist
for folder in [
    VIDEOS_DIR, METADATA_DIR, GPS_DIR,
    FRAMES_DIR, SELECTED_FRAMES_DIR, PREPROCESSED_DIR,
    SFM_DIR, DENSE_DIR, MODELS_DIR, REPORTS_DIR,
    UPLOADS_DIR, LOGS_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Frame Extraction Parameters
# ---------------------------------------------------------------------------
FRAME_EXTRACTION = {
    # Initial extraction pass — sample candidate pool from drone video
    "fps_sample_rate": 4.0,       # Extract 4 frames per second (candidate pool)
    "frame_interval": 8,           # Fallback interval if fps calculation not used
    "max_frames": 700,             # Upper bound for initial candidate pool
    "min_frames": 30,              # Minimum frames required for multi-view SfM
    "target_width": 1920,          # Full resolution for COLMAP (resize only if larger)
    "target_height": 1080,         # Full resolution for COLMAP
    # Outro / end-screen detection thresholds
    "outro_dark_mean_threshold": 60.0,    # Frames with mean luminance < this are dark/fade
    "outro_dark_consecutive": 5,          # Consecutive dark frames = outro detected
    "outro_search_tail_fraction": 0.15,   # Search last 15% of video for outro
}

# ---------------------------------------------------------------------------
# Keyframe Selection Parameters — intelligent selection from candidate pool
# ---------------------------------------------------------------------------
KEYFRAME_SELECTION = {
    "target_keyframes": 160,       # Optimal keyframe count for continuous orbital SfM & dense MVS
    "min_keyframes": 30,           # Absolute minimum to allow COLMAP to proceed
    "max_keyframes": 200,          # Hard cap
    # Sharpness filtering
    "blur_threshold": 40.0,        # Laplacian variance minimum floor
    "adaptive_blur_percentile": 15, # Use 15th-percentile sharpness as adaptive threshold
    # Brightness filtering
    "min_brightness": 30.0,        # Minimum average luminance
    "max_brightness": 240.0,       # Maximum average luminance
    # Motion / parallax
    "min_motion_magnitude": 0.6,   # Minimum optical flow magnitude between samples
    "max_motion_magnitude": 45.0,  # Maximum — very fast motion = motion blur, skip
    # Near-duplicate suppression (strictly identifies stationary hovering frames)
    "min_histogram_distance": 0.015, # Histogram L1-distance threshold
    "min_frame_spacing": 2,         # Minimum video frames between accepted keyframes
    "duplicate_ssim_threshold": 0.985, # Structural similarity threshold for duplicate detection
    "duplicate_hist_threshold": 0.995, # Histogram correlation threshold for duplicate detection
    # Sky / building heuristic
    "sky_upper_fraction": 0.25,    # Upper 25% of frame is considered sky zone
}

# ---------------------------------------------------------------------------
# Image Preprocessing Parameters
# ---------------------------------------------------------------------------
FRAME_QUALITY = {
    "blur_threshold": 40.0,               # Kept for backward-compatibility
    "min_brightness": 30.0,
    "max_brightness": 240.0,
    "duplicate_ssim_threshold": 0.985,
    "duplicate_hist_threshold": 0.995,
    "min_histogram_distance": 0.015,
}

IMAGE_PREPROCESSING = {
    "enable_clahe": True,           # Contrast Limited Adaptive Histogram Equalization
    "clahe_clip_limit": 2.0,
    "clahe_grid_size": (8, 8),
    "enable_denoising": True,
    "enable_dynamic_object_masking": True,   # AI-based human/vehicle/moving object mask
    "enable_sky_mask": True,        # Soft-suppress sky features during preprocessing
    "sky_mask_strength": 0.5,       # 0 = no suppression, 1 = full suppression
}

# ---------------------------------------------------------------------------
# Built-in SfM (fallback engine) Parameters
# ---------------------------------------------------------------------------
RECONSTRUCTION = {
    "feature_type": "SIFT",        # SIFT, ORB, or AKAZE
    "max_features": 8192,
    "match_ratio": 0.75,            # Lowe's ratio test threshold
    "min_inliers": 25,              # Minimum RANSAC inliers to accept two-view geometry
    "focal_length_prior_factor": 1.2, # Prior focal length = max(w, h) * factor
    "dense_downsample_voxel": 0.02,
    "outlier_nb_neighbors": 20,
    "outlier_std_ratio": 2.5,
    "poisson_depth": 9,             # Octree depth for Poisson surface reconstruction
    "ball_pivoting_radii": [0.02, 0.04, 0.08]
}

# ---------------------------------------------------------------------------
# COLMAP Execution Parameters (used by colmap_pipeline.py)
# ---------------------------------------------------------------------------
COLMAP_PARAMS = {
    # Feature extraction
    "camera_model": "OPENCV",       # Full radial + tangential distortion model
    "single_camera": True,          # CRITICAL: Enforce shared intrinsics for all images
    "sift_max_features": 8192,      # High density for building facades and architectural details
    "sift_estimate_affine_shape": False,
    "sift_domain_size_pooling": False,

    # Sequential matching (optimal for drone video)
    "sequential_overlap": 25,       # Match each frame to ±25 neighbors
    "sequential_quadratic_overlap": True,  # Quadratic overlap for orbital paths
    "sequential_loop_detection": False,    # Disabled unless vocab tree is provided

    # Mapper / SfM
    "mapper_init_min_tri_angle": 2.0,      # Lowered for drone video — orbital paths have small baselines
    "mapper_ba_refine_focal_length": True,  # Refine focal length in BA
    "mapper_ba_refine_principal_point": False,  # Don't refine PP (unstable without targets)
    "mapper_ba_refine_extra_params": True,  # Refine distortion coefficients
    "mapper_multiple_models": False,        # Force single model output
    "mapper_min_num_matches": 15,           # Minimum matches per image pair

    # Geometric verification
    "geo_verification_min_inliers": 15,
}

# ---------------------------------------------------------------------------
# Sparse Reconstruction Validation Thresholds
# ---------------------------------------------------------------------------
VALIDATION = {
    "min_registration_pct": 60.0,   # Minimum % of input images that must be registered
    "max_reprojection_error": 2.5,  # Maximum mean reprojection error (pixels)
    "min_sparse_points": 500,       # Minimum 3D sparse points for a valid model
    "min_track_length": 3.0,        # Minimum average track length
    "max_bounding_box_elongation": 15.0,  # Max ratio of longest:shortest axis
    # Camera trajectory smoothness: max acceptable position jump between consecutive cameras
    "max_cam_position_jump_factor": 8.0,  # Relative to median inter-camera spacing
}

# ---------------------------------------------------------------------------
# Open3D Point Cloud Cleaning Parameters
# ---------------------------------------------------------------------------
POINT_CLOUD_CLEANING = {
    "statistical_nb_neighbors": 20,
    "statistical_std_ratio": 2.5,   # Preserves real building surfaces while stripping isolated floating noise
    "radius_nb_points": 6,          # Minimum points in radius sphere
    "radius_radius_factor": 3.0,    # Radius = factor * avg nearest neighbor distance
    "voxel_downsample": False,      # Disabled by default to preserve detail
    "voxel_size": 0.05,
}

# ---------------------------------------------------------------------------
# COLMAP Discovery
# ---------------------------------------------------------------------------
CUSTOM_COLMAP_PATH = os.environ.get("COLMAP_PATH", "")


def get_colmap_executable():
    """Locate COLMAP executable in system PATH or environment."""
    if CUSTOM_COLMAP_PATH and os.path.exists(CUSTOM_COLMAP_PATH):
        return CUSTOM_COLMAP_PATH
    system_colmap = shutil.which("colmap")
    if system_colmap:
        return system_colmap
    # Common Windows install locations — prefer direct bin\colmap.exe
    common_paths = [
        r"C:\COLMAP\bin\colmap.exe",
        r"C:\COLMAP\COLMAP.bat",
        r"C:\COLMAP_NOCUDA\bin\colmap.exe",
        r"C:\COLMAP_NOCUDA\COLMAP.bat",
        r"C:\Program Files\COLMAP\colmap.exe",
        os.path.expanduser(r"~\COLMAP\colmap.exe")
    ]
    for p in common_paths:
        if os.path.exists(p):
            return p
    return None


COLMAP_EXECUTABLE = get_colmap_executable()
COLMAP_AVAILABLE = COLMAP_EXECUTABLE is not None

def get_colmap_env():
    """Construct environment with COLMAP bin and plugins on PATH for DLL resolution."""
    env = os.environ.copy()
    colmap_dir = Path(COLMAP_EXECUTABLE).parent if COLMAP_EXECUTABLE else Path("C:/COLMAP/bin")
    plugins_dir = colmap_dir.parent / "plugins"
    extra_paths = [str(colmap_dir), str(plugins_dir)]
    env["PATH"] = ";".join(extra_paths) + ";" + env.get("PATH", "")
    return env

COLMAP_ENV = get_colmap_env()
