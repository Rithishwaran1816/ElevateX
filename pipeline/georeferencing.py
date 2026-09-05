"""
Georeferencing & Spatial Alignment Module.
Processes GPS coordinates and flight metadata, aligns photogrammetric models
with WGS84 / UTM reference frames, or clearly records local coordinate fallback.
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import config
from services.gps_service import GPSService

class Georeferencer:
    def __init__(self, reports_dir: Path = config.REPORTS_DIR):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def process_georeferencing(self,
                               telemetry_data: Optional[List[Dict[str, Any]]] = None,
                               sparse_points_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Georeference reconstruction against flight telemetry.
        If telemetry is missing or lacks coordinates, explicit local-coordinate status is returned.
        """
        if not telemetry_data or len(telemetry_data) == 0:
            result = {
                "status": "UNAVAILABLE",
                "georeferencing_status": "LOCAL_COORDINATE_SYSTEM",
                "message": "Georeferencing data unavailable.",
                "reason": "GPS coordinates or synchronized flight metadata were not provided.",
                "result_coordinate_space": "Model reconstructed in local coordinate space (Arbitrary metric scale).",
                "reference_datum": "None (Local Euclidean)",
                "utm_zone": None,
                "anchor_point": None,
                "spatial_bounds_meters": None
            }
            self._save_report(result)
            return result

        # Analyze GPS telemetry via GPSService
        plane_info = GPSService.build_local_tangent_plane(telemetry_data)

        if not plane_info.get("available", False):
            result = {
                "status": "UNAVAILABLE",
                "georeferencing_status": "LOCAL_COORDINATE_SYSTEM",
                "message": "Georeferencing data unavailable.",
                "reason": plane_info.get("reason", "GPS coordinates could not be parsed."),
                "result_coordinate_space": "Model reconstructed in local coordinate space.",
                "reference_datum": "None (Local Euclidean)",
                "utm_zone": None,
                "anchor_point": None
            }
            self._save_report(result)
            return result

        anchor = plane_info["reference_anchor"]
        local_coords = plane_info["local_coordinates"]
        
        xs = [pt["x"] for pt in local_coords]
        ys = [pt["y"] for pt in local_coords]
        zs = [pt["z"] for pt in local_coords]

        bounds = {
            "min_x_east_m": round(float(np.min(xs)), 2),
            "max_x_east_m": round(float(np.max(xs)), 2),
            "min_y_north_m": round(float(np.min(ys)), 2),
            "max_y_north_m": round(float(np.max(ys)), 2),
            "min_z_up_m": round(float(np.min(zs)), 2),
            "max_z_up_m": round(float(np.max(zs)), 2),
            "area_span_m2": round(float((np.max(xs) - np.min(xs)) * (np.max(ys) - np.min(ys))), 2)
        }

        # 4x4 Georeferencing Transformation Matrix to UTM
        # Converts local model (x, y, z, 1) -> (UTM Easting, UTM Northing, Altitude, 1)
        transform_matrix = [
            [1.0, 0.0, 0.0, float(anchor["utm_easting"])],
            [0.0, 1.0, 0.0, float(anchor["utm_northing"])],
            [0.0, 0.0, 1.0, float(anchor["altitude"])],
            [0.0, 0.0, 0.0, 1.0]
        ]

        result = {
            "status": "SUCCESS",
            "georeferencing_status": "GEOREFERENCED_WGS84_UTM",
            "message": "Georeferencing applied successfully using synchronized flight telemetry.",
            "datum": "WGS84",
            "projection": "Universal Transverse Mercator (UTM)",
            "utm_zone": plane_info["utm_zone"],
            "anchor_wgs84": {
                "latitude": round(anchor["latitude"], 7),
                "longitude": round(anchor["longitude"], 7),
                "altitude_ellipsoidal_m": round(anchor["altitude"], 2)
            },
            "anchor_utm": {
                "easting": round(anchor["utm_easting"], 2),
                "northing": round(anchor["utm_northing"], 2)
            },
            "flight_trajectory_distance_meters": plane_info["flight_trajectory_length_meters"],
            "gps_waypoints_sampled": plane_info["sampled_waypoints_count"],
            "spatial_bounding_box": bounds,
            "transform_matrix_4x4": transform_matrix
        }

        self._save_report(result)
        return result

    def _save_report(self, report: Dict[str, Any]):
        out_file = self.reports_dir / "georeference_report.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
