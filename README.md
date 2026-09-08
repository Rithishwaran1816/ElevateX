# ElevateX — AI-Enabled Single-Pass Drone Video to Georeferenced 3D Model Generation System

**Smart India Hackathon (SIH) Problem Statement**: `SIH26158 – Single-Pass Drone Video to Accurate 3D Model Generation System`  
**Theme**: `Robotics and Drones`

---

## 1. Project Overview

**ElevateX** is an end-to-end photogrammetry, Computer Vision, and 3D geospatial reconstruction platform. It reconstructs dense 3D terrain, infrastructure, buildings, roads, and vegetation from **a single continuous UAV flight video** alongside optional GPS / telemetry metadata.

Unlike basic wrappers or mock generators, ElevateX uses **real photogrammetric algorithms** (Laplacian variance filtering, SIFT/ORB feature matching, Essential matrix RANSAC camera recovery, multi-view triangulation, dense disparity/optical flow point cloud synthesis, Open3D Poisson surface reconstruction, vertex color baking, WGS84 UTM georeferencing, and Three.js 3D WebGL visualization).

ElevateX seamlessly auto-detects **COLMAP** when available, and provides a built-in native **OpenCV + Open3D Photogrammetry Engine** so that the entire pipeline executes reliably on any standard development machine.

---

## 2. System Architecture

```text
                             SINGLE-PASS DRONE VIDEO
                                       │
                                       ▼
                       STAGE 1: VIDEO INGESTION & SAMPLING
                                       │
                                       ▼
                    STAGE 2: FRAME QUALITY ASSESSMENT & CLAHE
                     ├── Laplacian Variance Blur Scoring (> 80.0)
                     ├── Histogram & Perceptual Deduplication
                     ├── Over/Underexposure Filtering (35-235)
                     └── CLAHE Adaptive Illumination Recovery
                                       │
                                       ▼
                                       
                       STAGE 3: 3D PHOTOGRAMMETRIC RECONSTRUCTION
                     ├── SIFT / ORB Multi-Scale Feature Extraction
                     ├── FLANN / KD-Tree Ratio Matching (Lowe's Test)
                     ├── Essential Matrix Pose Recovery (5-Pt RANSAC)
                     ├── Cheirality-Verified Multi-View Triangulation
                     ├── Dense Multi-View Disparity Depth Synthesis
                     └── Open3D Statistical Outlier Removal
                                       │
                                       ▼
                       STAGE 4: SURFACE MESH GENERATION
                     ├── Normal Estimation & Consistent Orientation
                     ├── Screened Poisson Reconstruction (Depth 8)
                     └── Ball-Pivoting Algorithm (BPA) Fallback
                                       │
                                       ▼
                       STAGE 5: TEXTURE BAKING & 3D EXPORT
                     ├── Vertex Color Interpolation from Dense Cloud
                     └── Multi-Format Export: PLY, OBJ, Browser GLB
                                       │
                                       ▼
                       STAGE 6: GEOREFERENCING & SPATIAL ALIGNMENT
                     ├── DJI SRT / CSV / JSON Telemetry Parsing
                     ├── WGS84 Geodetic to UTM (Zone/Hemisphere)
                     └── Local Tangent Plane (ENU) Transformation Matrix
                                       │
                                       ▼
                       STAGE 7: QUALITY ASSESSMENT REPORT
                     └── Exports quality_report.json & georeference_report.json
                                       │
                                       ▼
                     INTERACTIVE WEB VIEWER (THREE.JS)
                     ├── Orbit, Pan, Zoom, Rotate Controls
                     ├── Solid / Wireframe / Point Cloud Toggles
                     ├── Point-to-Point 3D Measurement Tool (Ruler)
                     └── Telemetry & Sensor Availability Dashboard
```

---

## 3. Technologies Used

* **Core Backend**: Python 3.11+, Flask 3.1+
* **Computer Vision**: OpenCV (SIFT, ORB, FLANN, Essential Matrix, RANSAC, Optical Flow, CLAHE)
* **3D & Photogrammetry**: Open3D (Point Clouds, Poisson Surface Reconstruction, Outlier Removal), Trimesh (GLB/OBJ packaging), SciPy, NumPy, Pandas
* **Photogrammetry Engine**: COLMAP CLI integration + Built-in OpenCV/Open3D SfM
* **Frontend**: HTML5, CSS3 (Glassmorphism Dark Theme), Modern JavaScript
* **3D Visualization**: Three.js (WebGL, OrbitControls, GLTFLoader, PLYLoader, Raycasting Measurement Ruler)

---

## 4. Project Directory Structure

```text
elevate x/
├── app.py                      # Flask Web Application & REST API Server
├── config.py                   # Central system configuration & tool paths
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation & guide
├── run_all.py                  # Master automated CLI runner & demo generator
├── run_viewer.py               # Standalone 3D viewer launcher
│
├── data/
│   ├── input/
│   │   ├── videos/             # Source drone flight videos (MP4, MOV, AVI)
│   │   ├── metadata/           # Flight telemetry (CSV, JSON, DJI SRT)
│   │   └── gps/                # GPS logs
│   └── output/
│       ├── frames/             # Raw extracted video frames
│       ├── selected_frames/    # Quality-filtered clean frames
│       ├── preprocessed_frames/# CLAHE illumination-normalized frames
│       ├── sfm/                # Sparse reconstruction data
│       ├── dense/              # Dense multi-view stereo data
│       ├── models/             # Output 3D models (model.glb, PLY, OBJ)
│       └── reports/            # quality_report.json & georeference_report.json
│
├── pipeline/
│   ├── run_stage1.py           # Stage 1 standalone runner
│   ├── run_stage2.py           # Stage 2 standalone runner
│   ├── run_stage3.py           # Stage 3 standalone runner
│   ├── run_stage4.py           # Stage 4 standalone runner
│   ├── video_processing.py     # Video validation & metadata extraction
│   ├── frame_extraction.py     # Motion-adaptive intelligent frame extraction
│   ├── frame_quality.py        # Laplacian blur, brightness & duplicate filter
│   ├── image_preprocessing.py  # CLAHE & dynamic object masking interface
│   ├── colmap_pipeline.py      # COLMAP CLI orchestrator & diagnostic checker
│   ├── reconstruction.py       # Multi-view SfM & dense point cloud engine
│   ├── mesh_generation.py      # Poisson & Ball-Pivoting surface reconstructor
│   ├── texture_generation.py   # Vertex color baking & GLB/OBJ/PLY exporter
│   ├── georeferencing.py       # WGS84/UTM coordinate transformer
│   └── quality_assessment.py   # Quality metrics & compliance report generator
│
├── services/
│   ├── metadata_service.py     # Telemetry & EXIF parser
│   ├── gps_service.py          # Haversine, WGS84, and UTM geodesy service
│   └── processing_service.py   # Background asynchronous stage manager
│
├── web/
│   ├── templates/index.html    # Dashboard & Three.js WebGL viewer UI
│   └── static/
│       ├── css/style.css       # Clean dark theme styling
│       └── js/viewer.js        # Three.js 3D viewer & measurement ruler
│
└── tests/
    ├── test_frame_quality.py   # Unit tests for blur, exposure & deduplication
    ├── test_metadata.py        # Unit tests for telemetry & UTM math
    └── test_pipeline.py        # Integration tests for mesh, texture & georeferencing
```

---

## 5. Installation & Setup

### Step 1: Clone or Open Project
```bash
cd "c:\Users\rithishwaran\OneDrive\Desktop\elevate x"
```

### Step 2: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 3 (Optional): COLMAP Setup
COLMAP is optional. If installed on your system or placed in `C:\Program Files\COLMAP\colmap.exe`, ElevateX will automatically detect and utilize it.  
To specify a custom path, set the environment variable:
```bash
set COLMAP_PATH=C:\path\to\colmap.exe
```
If COLMAP is not present, ElevateX automatically executes the built-in pure-Python/OpenCV/Open3D Structure from Motion & Dense Photogrammetry Engine with zero interruption.

---

## 6. How to Run

### Option A: Complete Master CLI Pipeline
```bash
python run_all.py
```
* If no video is present in `data/input/videos/`, `run_all.py` automatically generates a synthetic orbital drone flight video with synchronized GPS telemetry for end-to-end verification.
* To provide custom drone video & telemetry:
```bash
python run_all.py --video "path/to/flight.mp4" --telemetry "path/to/telemetry.csv"
```

### Option B: Interactive Web Dashboard & 3D Viewer
```bash
python app.py
```
or
```bash
python run_viewer.py
```
Open your browser at: **`http://127.0.0.1:5000`**

From the Web Dashboard:
1. Drag and drop your drone video (`.mp4`, `.mov`, `.avi`) and optional flight telemetry (`.csv`, `.json`, `.srt`).
2. Click **"Upload & Inspect"** to view real-time video resolution, framerate, and sensor availability (GPS, IMU, RTK, Intrinsics).
3. Click **"Execute 3D Reconstruction"** to watch the 7-stage pipeline execute with live progress updates.
4. Interact with the reconstructed 3D model in Three.js (rotate, zoom, inspect wireframe/points, and measure real-world distances with the 3D ruler).

### Option C: Run Modular Individual Stages
* **Stage 1 (Video Ingestion & Extraction)**: `python pipeline/run_stage1.py --video "data/input/videos/drone_flight_sample.mp4"`
* **Stage 2 (Quality Filtering & CLAHE)**: `python pipeline/run_stage2.py --blur_threshold 80.0`
* **Stage 3 (SfM & Dense Cloud)**: `python pipeline/run_stage3.py`
* **Stage 4 (Mesh, Texture & Georeference)**: `python pipeline/run_stage4.py --telemetry "data/input/metadata/flight_telemetry_sample.csv"`

---

## 7. Running Unit & Integration Tests

Run the test suite with `pytest`:
```bash
pytest tests/ -v
```

---

## 8. Output Artifacts

All outputs are saved to `data/output/`:
* `data/output/models/model.glb`: Optimized binary GLTF 3D model with embedded vertex colors.
* `data/output/models/textured_model.ply`: High-resolution polygon mesh with RGB vertex colors.
* `data/output/models/textured_model.obj`: Standard wavefront OBJ 3D model.
* `data/output/models/dense_points_clean.ply`: Denoised multi-view dense point cloud.
* `data/output/models/sparse_points.ply`: Triangulated sparse tie-point cloud.
* `data/output/reports/quality_report.json`: Formal quality metrics, completeness %, and point counts.
* `data/output/reports/georeference_report.json`: WGS84 anchor coordinates, UTM projection, and 4x4 transformation matrix.

---

## 9. Limitations & Future AI Enhancements

* **Dynamic Object Handling**: Moving objects (vehicles, pedestrians) are flagged during sequential feature matching. Future versions integrate real-time YOLOv8 dynamic mask segmentation.
* **Occlusion & Missing Surface Detection**: Occluded angles are identified via ray coverage density and marked in the quality report without fabricating unseen geometry.
* **Future Radiance Modules**: ElevateX architecture is designed with clear extension interfaces for **3D Gaussian Splatting (3DGS)**, **Neural Radiance Fields (NeRF)**, and **Monocular Depth Priors**.

---

## 10. Hackathon Compliance Checklist

| SIH Requirement | Status | Implementation Details |
|---|---|---|
| Single-Pass Drone Video Ingestion | **IMPLEMENTED** | Supports MP4, MOV, AVI, MKV with frame rate sampling |
| Frame Quality & Blur Filtering | **IMPLEMENTED** | Laplacian variance, exposure bounds & histogram deduplication |
| Illumination Normalization | **IMPLEMENTED** | CLAHE adaptive luminance equalization |
| Feature Extraction & Matching | **IMPLEMENTED** | SIFT/ORB multi-scale detector with FLANN ratio matcher |
| Camera Pose & SfM | **IMPLEMENTED** | Essential matrix 5-point RANSAC & multi-view triangulation |
| Dense Point Cloud Generation | **IMPLEMENTED** | Multi-view disparity depth synthesis & statistical outlier filter |
| Watertight 3D Mesh Generation | **IMPLEMENTED** | Screened Poisson Reconstruction & Ball-Pivoting in Open3D |
| Texture & Vertex Coloring | **IMPLEMENTED** | Color projection & GLB/OBJ/PLY export |
| Georeferencing & Flight Telemetry | **IMPLEMENTED** | WGS84 to UTM transformation matrix & local tangent plane |
| Interactive Web 3D Viewer | **IMPLEMENTED** | Three.js WebGL viewer with OrbitControls & 3D measurement ruler |
| Sensor Availability Diagnostics | **IMPLEMENTED** | Live status indicators for GPS, IMU, RTK & Camera Intrinsics |
