"""
ElevateX Master Execution Script.
Validates environment, checks COLMAP, ingests drone video & telemetry,
executes the 7-stage reconstruction pipeline, and prints diagnostic results.
"""
import os
import sys
import time
import math
import argparse
import numpy as np
import cv2
import pandas as pd
from pathlib import Path

# Add workspace to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import config
from services.metadata_service import MetadataService
from services.gps_service import GPSService
from pipeline.video_processing import VideoProcessor
from pipeline.frame_extraction import FrameExtractor
from pipeline.frame_quality import FrameQualityAssessor
from pipeline.image_preprocessing import ImagePreprocessor
from pipeline.reconstruction import PhotogrammetryReconstructor
from pipeline.mesh_generation import MeshGenerator
from pipeline.texture_generation import TextureGenerator
from pipeline.georeferencing import Georeferencer
from pipeline.quality_assessment import QualityAssessor

def generate_sample_drone_data(output_video_path: Path, output_tel_path: Path):
    """
    Generate synthetic aerial drone footage and synchronized GPS flight telemetry
    for testing and verification on clean installations.
    """
    print(f"\n[DEMO GENERATOR] Creating synthetic drone flight footage and telemetry...")
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    output_tel_path.parent.mkdir(parents=True, exist_ok=True)

    w, h = 1280, 720
    fps = 30.0
    num_frames = 150 # 5-second orbital pass
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (w, h))

    # Base GPS Anchor (New Delhi / India Gate area)
    base_lat = 28.6129
    base_lon = 77.2295
    base_alt = 120.0 # meters AGL

    telemetry_records = []

    # Render a 3D terrain scene with textured cubes/buildings, roads and terrain
    for i in range(num_frames):
        theta = (i / num_frames) * (math.pi / 1.8) # ~100-degree orbit arc
        cam_x = 35.0 * math.cos(theta)
        cam_y = 35.0 * math.sin(theta)

        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Gradient terrain sky / ground
        frame[:int(h * 0.40)] = [210, 180, 140]  # Atmosphere
        frame[int(h * 0.40):] = [45, 115, 65]    # Lush terrain ground

        # Grid texture lines
        for gx in range(0, w, 50):
            cv2.line(frame, (gx, int(h * 0.40)), (int(gx + (gx - w/2) * 0.6), h), (35, 95, 55), 1)
        for gy in range(int(h * 0.40), h, 30):
            cv2.line(frame, (0, gy), (w, gy), (35, 95, 55), 1)

        # Draw multiple 3D Infrastructure Buildings with Parallax
        buildings = [
            {"offset_x": 0, "offset_y": 60, "bw": 200, "bh": 160, "col_front": (160, 175, 190), "col_roof": (200, 215, 230)},
            {"offset_x": -260, "offset_y": 90, "bw": 140, "bh": 110, "col_front": (140, 155, 175), "col_roof": (180, 195, 215)},
            {"offset_x": 260, "offset_y": 80, "bw": 150, "bh": 130, "col_front": (150, 165, 180), "col_roof": (190, 205, 220)}
        ]

        for b in buildings:
            center_x = int(w / 2 + b["offset_x"] + math.sin(theta) * 160)
            center_y = int(h / 2 + b["offset_y"])
            bw, bh = b["bw"], b["bh"]

            # Front facade
            pts_front = np.array([
                [center_x - bw//2, center_y],
                [center_x + bw//2, center_y],
                [center_x + bw//2, center_y - bh],
                [center_x - bw//2, center_y - bh]
            ], np.int32)
            cv2.fillPoly(frame, [pts_front], b["col_front"])
            cv2.polylines(frame, [pts_front], True, (30, 40, 50), 2)

            # Roof with perspective shift
            roof_offset_x = int(math.cos(theta) * 55)
            roof_offset_y = -35
            pts_roof = np.array([
                [center_x - bw//2, center_y - bh],
                [center_x + bw//2, center_y - bh],
                [center_x + bw//2 + roof_offset_x, center_y - bh + roof_offset_y],
                [center_x - bw//2 + roof_offset_x, center_y - bh + roof_offset_y]
            ], np.int32)
            cv2.fillPoly(frame, [pts_roof], b["col_roof"])
            cv2.polylines(frame, [pts_roof], True, (30, 40, 50), 2)

            # Side facade
            pts_side = np.array([
                [center_x + bw//2, center_y],
                [center_x + bw//2 + roof_offset_x, center_y + roof_offset_y],
                [center_x + bw//2 + roof_offset_x, center_y - bh + roof_offset_y],
                [center_x + bw//2, center_y - bh]
            ], np.int32)
            cv2.fillPoly(frame, [pts_side], (110, 125, 140))
            cv2.polylines(frame, [pts_side], True, (30, 40, 50), 2)

            # High-frequency textured window grid for SIFT feature detector
            for wy in range(center_y - bh + 15, center_y - 15, 25):
                for wx in range(center_x - bw//2 + 18, center_x + bw//2 - 18, 28):
                    cv2.rectangle(frame, (wx, wy), (wx + 14, wy + 14), (40, 75, 130), -1)
                    cv2.rectangle(frame, (wx, wy), (wx + 14, wy + 14), (245, 245, 245), 1)

        # Realistic subtle sensor noise
        noise = np.random.normal(0, 3, frame.shape).astype(np.uint8)
        frame = cv2.add(frame, noise)

        out.write(frame)

        # Synchronized GPS telemetry waypoint
        d_lat = (cam_y / 111139.0)
        d_lon = (cam_x / (111139.0 * math.cos(math.radians(base_lat))))
        telemetry_records.append({
            "timestamp_sec": round(i / fps, 3),
            "latitude": round(base_lat + d_lat, 7),
            "longitude": round(base_lon + d_lon, 7),
            "altitude": round(base_alt + (i * 0.08), 2),
            "pitch": round(-25.0 + math.sin(theta)*6, 2),
            "roll": round(math.cos(theta)*4, 2),
            "yaw": round(math.degrees(theta), 2),
            "rtk_status": "FIXED_RTK"
        })

    out.release()

    # Save telemetry CSV
    df = pd.DataFrame(telemetry_records)
    df.to_csv(output_tel_path, index=False)
    print(f"[DEMO GENERATOR] Video saved to: {output_video_path}")
    print(f"[DEMO GENERATOR] Telemetry saved to: {output_tel_path}")


def main():
    parser = argparse.ArgumentParser(description="ElevateX Photogrammetry Pipeline Runner")
    parser.add_argument("--video", type=str, default=None, help="Input drone video path")
    parser.add_argument("--telemetry", type=str, default=None, help="Input telemetry CSV/JSON/SRT path")
    parser.add_argument("--force_builtin", action="store_true", help="Force built-in SfM engine")
    args = parser.parse_args()

    print("================================================================================")
    print("  ELEVATEX — AI-Enabled Single-Pass Drone 3D Model Generation System")
    print("  Problem Statement: SIH26158 | Theme: Robotics and Drones")
    print("================================================================================")

    # 1. Environment and Tool Validation
    print("\n--- [ENVIRONMENT VALIDATION] ---")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"OpenCV: {cv2.__version__}")
    try:
        import open3d as o3d
        print(f"Open3D: {o3d.__version__}")
    except ImportError:
        print("Open3D: NOT INSTALLED")

    colmap_info = "AVAILABLE (" + str(config.COLMAP_EXECUTABLE) + ")" if config.COLMAP_AVAILABLE else "NOT FOUND (Will use built-in OpenCV/Open3D Photogrammetry Engine)"
    print(f"COLMAP Photogrammetry: {colmap_info}")

    # 2. Input Resolution
    video_path = args.video
    telemetry_path = args.telemetry

    if not video_path:
        # Check if video exists in input directory
        existing_videos = list(config.VIDEOS_DIR.glob("*.mp4")) + list(config.VIDEOS_DIR.glob("*.mov"))
        if existing_videos:
            video_path = str(existing_videos[0])
            print(f"Found existing input video: {video_path}")
        else:
            # Generate sample drone footage for immediate verification
            sample_vid = config.VIDEOS_DIR / "drone_flight_sample.mp4"
            sample_tel = config.METADATA_DIR / "flight_telemetry_sample.csv"
            generate_sample_drone_data(sample_vid, sample_tel)
            video_path = str(sample_vid)
            telemetry_path = str(sample_tel)

    if not telemetry_path:
        existing_tel = list(config.METADATA_DIR.glob("*.csv")) + list(config.METADATA_DIR.glob("*.json")) + list(config.METADATA_DIR.glob("*.srt"))
        if existing_tel:
            telemetry_path = str(existing_tel[0])

    start_time = time.time()

    # -------------------------------------------------------------
    # Stage 1: Video Ingestion & Frame Extraction
    # -------------------------------------------------------------
    print("\n>>> [STAGE 1/7] VIDEO INGESTION & FRAME EXTRACTION")
    v_proc = VideoProcessor(video_path)
    v_res = v_proc.validate_and_inspect()
    if not v_res["valid"]:
        print(f"[FATAL] Video error: {v_res.get('error')}")
        sys.exit(1)

    video_meta = v_res["metadata"]
    print(f" -> Resolution: {video_meta['resolution']} @ {video_meta['fps']} FPS | Duration: {video_meta['duration_formatted']}")

    extractor = FrameExtractor()
    extract_res = extractor.extract_frames(video_path)
    print(f" -> Candidate frames extracted : {extract_res['extracted_frames_count']}")
    print(f" -> Useful footage ends at     : frame {extract_res.get('outro_start_frame', '?')} "
          f"({extract_res.get('useful_video_frames', '?')} frames kept)")
    print(f" -> Pre-rejected (blur/dark)   : {extract_res.get('pre_rejected_blur', 0)} blur, "
          f"{extract_res.get('pre_rejected_dark', 0)} dark")

    # -------------------------------------------------------------
    # Stage 2: Quality Assessment & Preprocessing
    # -------------------------------------------------------------
    print("\n>>> [STAGE 2/7] FRAME QUALITY FILTERING & PREPROCESSING")
    qa_filter = FrameQualityAssessor()
    frame_qa = qa_filter.assess_and_filter()
    print(f" -> Keyframes selected          : {frame_qa['accepted_frames']} / {frame_qa['total_frames']} candidates")
    print(f" -> Rejected (blur)            : {frame_qa['rejected_blur']}")
    print(f" -> Rejected (exposure)        : {frame_qa.get('rejected_exposure', 0)}")
    print(f" -> Rejected (duplicates)      : {frame_qa['rejected_duplicates']}")
    print(f" -> Average sharpness (Lap var): {frame_qa['average_sharpness']}")
    print(f" -> Average motion magnitude   : {frame_qa.get('average_motion_magnitude', 'N/A')}")
    print(f" -> Blur threshold used        : {frame_qa.get('blur_threshold_used', 'N/A')}")

    preprocessor = ImagePreprocessor()
    prep_res = preprocessor.preprocess_images()
    print(f" -> Preprocessing applied to   : {prep_res['preprocessed_count']} frames")
    print(f"    CLAHE={prep_res['clahe_applied']} | Denoise={prep_res['denoising_applied']} | SkyMask={prep_res.get('sky_mask_applied', False)}")

    # -------------------------------------------------------------
    # Stage 3: SfM & Dense Reconstruction
    # -------------------------------------------------------------
    print("\n>>> [STAGE 3/7] STRUCTURE FROM MOTION & DENSE MVS")
    reconstructor = PhotogrammetryReconstructor()
    rec_res = reconstructor.run_reconstruction(force_builtin=args.force_builtin)
    print(f" -> Engine Used              : {rec_res.get('engine')}")
    print(f" -> Registered Images        : {rec_res.get('registered_images', 'N/A')} / {rec_res.get('total_images', 'N/A')}")
    print(f" -> Registration Percentage  : {rec_res.get('registration_pct', 0.0):.1f}%")
    print(f" -> Mean Reprojection Error  : {rec_res.get('mean_reprojection_error', 0.0):.3f} px")
    print(f" -> Mean Track Length        : {rec_res.get('mean_track_length', 0.0):.2f}")
    print(f" -> Sparse Points            : {rec_res.get('sparse_points_count', 0):,}")
    print(f" -> Dense Points             : {rec_res.get('dense_points_count', 0):,}")
    print(f" -> Validation Result        : {rec_res.get('validation_result', 'N/A')}")
    print(f" -> Sparse PLY               : {rec_res.get('sparse_ply')}")
    print(f" -> Dense PLY (clean)        : {rec_res.get('clean_dense_ply') or rec_res.get('dense_ply')}")
    print("")
    for diag in rec_res.get('diagnostic_messages', []):
        print(f"    [DIAG] {diag}")

    # -------------------------------------------------------------
    # Stage 4: Mesh Surface Generation
    # -------------------------------------------------------------
    print("\n>>> [STAGE 4/7] 3D MESH GENERATION")
    mesh_gen = MeshGenerator()
    mesh_res = mesh_gen.generate_mesh()
    print(f" -> Method: {mesh_res['method_used']} Surface Reconstruction")
    print(f" -> Vertices: {mesh_res['vertices_count']:,} | Triangles: {mesh_res['faces_count']:,}")
    print(f" -> Mesh Output: {mesh_res['mesh_ply']}")

    # -------------------------------------------------------------
    # Stage 5: Texture Generation & Export
    # -------------------------------------------------------------
    print("\n>>> [STAGE 5/7] TEXTURE GENERATION & WEB 3D EXPORT")
    tex_gen = TextureGenerator()
    tex_res = tex_gen.generate_textured_model(mesh_path=mesh_res['mesh_ply'])
    print(f" -> Exported Formats: {', '.join(tex_res['export_formats'])}")
    if tex_res.get('model_glb'):
        print(f" -> Browser GLB: {tex_res['model_glb']}")

    # -------------------------------------------------------------
    # Stage 6: Georeferencing
    # -------------------------------------------------------------
    print("\n>>> [STAGE 6/7] GEOREFERENCING & SPATIAL ALIGNMENT")
    telemetry_data = []
    sensor_status = MetadataService.get_sensor_status_summary(None)
    if telemetry_path and Path(telemetry_path).exists():
        parsed_tel = MetadataService.parse_flight_telemetry(telemetry_path)
        telemetry_data = parsed_tel.get("summary", {}).get("telemetry_data", [])
        sensor_status = MetadataService.get_sensor_status_summary(parsed_tel)

    georef = Georeferencer()
    georef_res = georef.process_georeferencing(telemetry_data=telemetry_data)
    print(f" -> Status: {georef_res['georeferencing_status']}")
    print(f" -> Coordinate System: {georef_res.get('datum', 'Local Euclidean')}")
    if georef_res.get('utm_zone'):
        print(f" -> UTM Zone: {georef_res['utm_zone']} | Waypoints: {georef_res.get('gps_waypoints_sampled')}")

    # -------------------------------------------------------------
    # Stage 7: Quality Assessment Report
    # -------------------------------------------------------------
    print("\n>>> [STAGE 7/7] QUALITY & METRIC ASSESSMENT REPORT")
    qa = QualityAssessor()
    q_report = qa.generate_full_report(
        video_meta=video_meta,
        frame_quality_meta=frame_qa,
        reconstruction_meta=rec_res,
        mesh_meta=mesh_res,
        georeference_meta=georef_res,
        sensor_status=sensor_status,
        start_time_seconds=start_time
    )

    qa_status = q_report.get('quality_assessment', {}).get('quality_status', 'UNKNOWN')
    print(f"\n================================================================================")
    print(f"                         ELEVATEX EXECUTION SUMMARY                             ")
    print(f"================================================================================")
    print(f" Processing Time              : {q_report['processing_time_seconds']} seconds")
    print(f" Ingested Video               : {video_meta['filename']} ({video_meta['resolution']}, {video_meta['fps']} FPS)")
    print(f" Total Input Frames           : {q_report['frame_quality']['total_input_frames']}")
    print(f" Selected Keyframes           : {q_report['frame_quality']['selected_keyframes']}")
    print(f" Registered Images            : {q_report['reconstruction_metrics']['registered_images']} / {q_report['reconstruction_metrics']['total_input_images']}")
    print(f" Registration Percentage      : {q_report['reconstruction_metrics']['registration_percentage']:.1f}%")
    print(f" Mean Reprojection Error      : {q_report['reconstruction_metrics']['mean_reprojection_error_pixels']:.3f} px")
    print(f" Sparse Points                : {q_report['reconstruction_metrics']['sparse_points']:,}")
    print(f" Dense Points                 : {q_report['reconstruction_metrics']['dense_points']:,}")
    print(f" Mesh Vertices / Faces        : {q_report['reconstruction_metrics']['mesh_vertices']:,} / {q_report['reconstruction_metrics']['mesh_faces']:,}")
    print(f" RECONSTRUCTION QUALITY       : {qa_status}")
    print(f" Georeferencing Status        : {georef_res['georeferencing_status']}")
    print(f" Metric Accuracy Note         : {q_report['georeferencing']['metric_accuracy']}")
    print(f"")
    print(f" Quality Explanation:")
    for line in q_report.get('quality_assessment', {}).get('quality_explanation', '').split('. '):
        if line.strip():
            print(f"   {line.strip()}.")
    if q_report.get('quality_assessment', {}).get('diagnostic_messages'):
        print(f"\n Diagnostics:")
        for msg in q_report['quality_assessment']['diagnostic_messages']:
            print(f"   • {msg}")
    print(f"")
    print(f" Quality Report Saved         : data/output/reports/quality_report.json")
    print(f" 3D Model Saved               : data/output/models/model.glb")
    print(f"================================================================================")
    print(f"\nTo launch the interactive Web Viewer, run:\n  python run_viewer.py\nor\n  python app.py\n")

if __name__ == "__main__":
    main()
