"""
ElevateX Flask Web Application & REST API Server.
Provides endpoints for video/telemetry upload, pipeline execution,
live progress monitoring, 3D model streaming, and metric reporting.
"""
import os
import re
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory

import config
from services.processing_service import ProcessingService
from services.metadata_service import MetadataService

app = Flask(
    __name__,
    template_folder=str(config.BASE_DIR / "web" / "templates"),
    static_folder=str(config.BASE_DIR / "web" / "static")
)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1 GB upload limit

processor_service = ProcessingService()

@app.route("/")
def index():
    """Render main dashboard & 3D viewer."""
    return render_template("index.html")

@app.route("/api/system/status", methods=["GET"])
def system_status():
    """Get system environment and photogrammetry tool availability."""
    return jsonify({
        "status": "ONLINE",
        "colmap_available": config.COLMAP_AVAILABLE,
        "colmap_path": config.COLMAP_EXECUTABLE,
        "default_focal_prior": config.RECONSTRUCTION["focal_length_prior_factor"]
    })

@app.route("/api/upload", methods=["POST"])
def upload_files():
    """Upload drone video and optional flight telemetry / GPS file."""
    try:
        if "video" not in request.files:
            return jsonify({"success": False, "error": "No video file provided in the upload request"}), 400

        video_file = request.files["video"]
        if not video_file or video_file.filename == "":
            return jsonify({"success": False, "error": "Empty video filename"}), 400

        # Safe filename
        safe_v_name = re.sub(r"[^\w\.\-\_]", "_", video_file.filename)
        video_path = config.VIDEOS_DIR / safe_v_name
        video_file.save(str(video_path))

        telemetry_path = None
        if "telemetry" in request.files:
            tel_file = request.files["telemetry"]
            if tel_file and tel_file.filename != "":
                safe_t_name = re.sub(r"[^\w\.\-\_]", "_", tel_file.filename)
                tel_path = config.METADATA_DIR / safe_t_name
                tel_file.save(str(tel_path))
                telemetry_path = str(tel_path)

        # Inspect video metadata
        video_meta = MetadataService.extract_video_metadata(str(video_path))

        # Inspect sensor availability
        telemetry_summary = None
        if telemetry_path:
            tel_info = MetadataService.parse_flight_telemetry(telemetry_path)
            telemetry_summary = tel_info.get("summary")
            sensor_status = MetadataService.get_sensor_status_summary(tel_info)
        else:
            sensor_status = MetadataService.get_sensor_status_summary(None)

        return jsonify({
            "success": True,
            "video_path": str(video_path),
            "telemetry_path": telemetry_path,
            "video_metadata": video_meta,
            "telemetry_summary": telemetry_summary,
            "sensor_status": sensor_status
        })
    except Exception as e:
        return jsonify({"success": False, "error": f"Upload processing error: {str(e)}"}), 500

@app.route("/api/sample/load", methods=["POST"])
def load_sample_video():
    """Quick-load the verified demo drone flight video and telemetry."""
    sample_vid = config.VIDEOS_DIR / "drone_flight_sample.mp4"
    sample_tel = config.METADATA_DIR / "flight_telemetry_sample.csv"

    if not sample_vid.exists():
        from run_all import generate_sample_drone_data
        generate_sample_drone_data(sample_vid, sample_tel)

    video_meta = MetadataService.extract_video_metadata(str(sample_vid))
    tel_info = MetadataService.parse_flight_telemetry(str(sample_tel))
    sensor_status = MetadataService.get_sensor_status_summary(tel_info)

    return jsonify({
        "success": True,
        "video_path": str(sample_vid),
        "telemetry_path": str(sample_tel),
        "video_metadata": video_meta,
        "telemetry_summary": tel_info.get("summary"),
        "sensor_status": sensor_status
    })

@app.route("/api/process/start", methods=["POST"])
def start_processing():
    """Start asynchronous 7-stage reconstruction pipeline."""
    data = request.get_json() or {}
    video_path = data.get("video_path")
    telemetry_path = data.get("telemetry_path")

    if not video_path or not Path(video_path).exists():
        return jsonify({"success": False, "error": "Valid video_path is required"}), 400

    current = processor_service.get_status()
    if current["state"] == "RUNNING":
        return jsonify({"success": False, "error": "Reconstruction already in progress"}), 409

    processor_service.start_pipeline_async(video_path, telemetry_path)
    return jsonify({"success": True, "message": "ElevateX reconstruction started successfully."})

@app.route("/api/process/status", methods=["GET"])
def process_status():
    """Poll pipeline execution progress and stage states."""
    return jsonify(processor_service.get_status())

@app.route("/api/models/<path:filename>")
def serve_model(filename):
    """Serve generated 3D models (GLB, PLY, OBJ)."""
    return send_from_directory(str(config.MODELS_DIR), filename)

@app.route("/api/reports/<path:filename>")
def serve_report(filename):
    """Serve JSON reports."""
    return send_from_directory(str(config.REPORTS_DIR), filename)

def run_server(host="127.0.0.1", port=5000, debug=False):
    print(f"\n=======================================================")
    print(f"   ELEVATEX WEB APPLICATION RUNNING")
    print(f"   URL: http://{host}:{port}")
    print(f"   COLMAP Status: {'AVAILABLE (' + str(config.COLMAP_EXECUTABLE) + ')' if config.COLMAP_AVAILABLE else 'BUILT-IN SFM ENGINE'}")
    print(f"=======================================================\n")
    app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    run_server(host="0.0.0.0", port=5000, debug=False)
