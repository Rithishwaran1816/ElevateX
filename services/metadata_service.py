"""
Metadata Extraction & Sensor Discovery Service.
Parses drone video properties, telemetry files (CSV/JSON/SRT), and reports sensor availability.
"""
import os
import re
import json
import cv2
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List

class MetadataService:
    def __init__(self):
        pass

    @staticmethod
    def extract_video_metadata(video_path: str) -> Dict[str, Any]:
        """Extract intrinsic video parameters using OpenCV."""
        path_obj = Path(video_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video stream: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)]) if fourcc else "Unknown"

        # If width/height are 0 from headers, read a frame directly
        if width <= 0 or height <= 0:
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                height, width = test_frame.shape[:2]
                if total_frames <= 0:
                    total_frames = 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if fps <= 0:
            fps = 30.0

        duration_sec = total_frames / fps if fps > 0 else 0.0
        
        cap.release()

        return {
            "filename": path_obj.name,
            "filepath": str(path_obj.resolve()),
            "filesize_mb": round(path_obj.stat().st_size / (1024 * 1024), 2),
            "width": width,
            "height": height,
            "resolution": f"{width}x{height}",
            "fps": round(fps, 2),
            "total_frames": total_frames,
            "duration_seconds": round(duration_sec, 2),
            "duration_formatted": f"{int(duration_sec // 60)}m {int(duration_sec % 60)}s",
            "codec": codec.strip() or "H.264/MP4V"
        }

    @staticmethod
    def parse_flight_telemetry(file_path: str) -> Dict[str, Any]:
        """
        Parse drone flight telemetry from CSV, JSON, or DJI SRT files.
        """
        path = Path(file_path)
        if not path.exists():
            return {"status": "UNAVAILABLE", "error": f"Telemetry file {file_path} does not exist"}

        suffix = path.suffix.lower()
        records = []
        sensor_info = {
            "has_gps": False,
            "has_altitude": False,
            "has_imu": False,
            "has_rtk": False,
            "has_camera_intrinsics": False,
            "record_count": 0,
            "telemetry_data": []
        }

        try:
            if suffix == ".json":
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        records = data
                    elif isinstance(data, dict) and "records" in data:
                        records = data["records"]
                    elif isinstance(data, dict):
                        records = [data]

            elif suffix in [".csv", ".txt"]:
                df = pd.read_csv(path)
                records = df.to_dict(orient="records")

            elif suffix == ".srt":
                # Parse DJI SRT subtitle stream
                records = MetadataService._parse_dji_srt(str(path))

            if records:
                sensor_info["record_count"] = len(records)
                sample = records[0] if len(records) > 0 else {}
                sample_keys = [str(k).lower() for k in sample.keys()]

                sensor_info["has_gps"] = any(k in sample_keys for k in ["latitude", "lat", "gps_latitude", "latitude(deg)"])
                sensor_info["has_altitude"] = any(k in sample_keys for k in ["altitude", "alt", "rel_alt", "rel_altitude", "barometer", "baro"])
                sensor_info["has_imu"] = any(k in sample_keys for k in ["pitch", "roll", "yaw", "gyro_x", "accel_x", "gimbal_pitch"])
                sensor_info["has_rtk"] = any(k in sample_keys for k in ["rtk_flag", "rtk_status", "is_rtk", "fix_type", "diff_status"])
                sensor_info["has_camera_intrinsics"] = any(k in sample_keys for k in ["focal_length", "fx", "fy", "sensor_width_mm"])
                sensor_info["telemetry_data"] = records

            return {
                "status": "AVAILABLE" if len(records) > 0 else "EMPTY",
                "source_file": path.name,
                "summary": sensor_info
            }

        except Exception as e:
            return {"status": "ERROR", "error": str(e)}

    @staticmethod
    def _parse_dji_srt(srt_path: str) -> List[Dict[str, Any]]:
        """Extract GPS and telemetry from DJI embedded SRT logs."""
        results = []
        with open(srt_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        blocks = content.strip().split("\n\n")
        for block in blocks:
            lines = [line.strip() for line in block.split("\n") if line.strip()]
            if len(lines) < 2:
                continue

            entry = {}
            for line in lines:
                # Lat/Long extraction
                lat_match = re.search(r"\[latitude\s*:\s*([\-\d\.]+)\]|LATITUDE:\s*([\-\d\.]+)", line, re.IGNORECASE)
                lon_match = re.search(r"\[longitude\s*:\s*([\-\d\.]+)\]|LONGITUDE:\s*([\-\d\.]+)", line, re.IGNORECASE)
                alt_match = re.search(r"\[rel_alt\s*:\s*([\-\d\.]+)\]|\[altitude\s*:\s*([\-\d\.]+)\]|ALT:\s*([\-\d\.]+)", line, re.IGNORECASE)
                iso_match = re.search(r"\[iso\s*:\s*(\d+)\]|ISO:\s*(\d+)", line, re.IGNORECASE)
                shutter_match = re.search(r"\[shutter\s*:\s*([\w\/\.]+)\]|SHUTTER:\s*([\w\/\.]+)", line, re.IGNORECASE)

                if lat_match:
                    entry["latitude"] = float(lat_match.group(1) or lat_match.group(2))
                if lon_match:
                    entry["longitude"] = float(lon_match.group(1) or lon_match.group(2))
                if alt_match:
                    entry["altitude"] = float(alt_match.group(1) or alt_match.group(2) or alt_match.group(3))
                if iso_match:
                    entry["iso"] = int(iso_match.group(1) or iso_match.group(2))
                if shutter_match:
                    entry["shutter"] = shutter_match.group(1) or shutter_match.group(2)

            if "latitude" in entry and "longitude" in entry:
                results.append(entry)

        return results

    @staticmethod
    def get_sensor_status_summary(telemetry_info: Optional[Dict[str, Any]] = None, has_exif_gps: bool = False) -> Dict[str, str]:
        """
        Produce clear, standard sensor availability status dictionary.
        """
        has_gps = has_exif_gps
        has_flight_meta = False
        has_imu = False
        has_rtk = False
        has_intrinsics = False

        if telemetry_info and telemetry_info.get("status") == "AVAILABLE":
            has_flight_meta = True
            summary = telemetry_info.get("summary", {})
            has_gps = has_gps or summary.get("has_gps", False)
            has_imu = summary.get("has_imu", False)
            has_rtk = summary.get("has_rtk", False)
            has_intrinsics = summary.get("has_camera_intrinsics", False)

        return {
            "GPS": "AVAILABLE" if has_gps else "NOT AVAILABLE",
            "Flight Metadata": "AVAILABLE" if has_flight_meta else "NOT AVAILABLE",
            "IMU": "AVAILABLE" if has_imu else "NOT AVAILABLE",
            "RTK": "AVAILABLE" if has_rtk else "NOT AVAILABLE",
            "Camera Intrinsics": "AVAILABLE" if has_intrinsics else "NOT AVAILABLE"
        }
