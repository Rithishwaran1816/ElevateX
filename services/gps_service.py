"""
GPS & Geospatial Coordinates Service.
Implements WGS84 Geodesy, Local Tangent Plane (ENU / UTM) conversions,
and coordinate transformations for photogrammetric reconstruction.
"""
import math
import numpy as np
from typing import Tuple, List, Dict, Any, Optional

class GPSService:
    # WGS84 Ellipsoid constants
    WGS84_A = 6378137.0         # semi-major axis in meters
    WGS84_E2 = 0.00669437999014 # first eccentricity squared

    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great circle distance between two points in meters."""
        r = 6371000.0 # Earth radius in meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi / 2.0)**2 + \
            math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0)**2
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return r * c

    @staticmethod
    def latlon_to_utm(lat: float, lon: float) -> Tuple[float, float, int, str]:
        """
        Convert WGS84 Latitude and Longitude to UTM Easting and Northing (meters).
        Standard Transverse Mercator Projection.
        """
        zone_number = int((lon + 180) / 6) + 1
        hemisphere = 'N' if lat >= 0 else 'S'

        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        lon_origin = (zone_number - 1) * 6 - 180 + 3
        lon_origin_rad = math.radians(lon_origin)

        a = GPSService.WGS84_A
        e2 = GPSService.WGS84_E2
        e_prime2 = e2 / (1 - e2)
        n = a / math.sqrt(1 - e2 * math.sin(lat_rad)**2)
        t = math.tan(lat_rad)**2
        c = e_prime2 * math.cos(lat_rad)**2
        al = math.cos(lat_rad) * (lon_rad - lon_origin_rad)

        m = a * ((1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256) * lat_rad -
                 (3 * e2 / 8 + 3 * e2**2 / 32 + 45 * e2**3 / 1024) * math.sin(2 * lat_rad) +
                 (15 * e2**2 / 256 + 45 * e2**3 / 1024) * math.sin(4 * lat_rad) -
                 (35 * e2**3 / 3072) * math.sin(6 * lat_rad))

        easting = 500000.0 + 0.9996 * n * (al + (1 - t + c) * al**3 / 6 +
                                           (5 - 18 * t + t**2 + 72 * c - 58 * e_prime2) * al**5 / 120)
        
        northing = 0.9996 * (m + n * math.tan(lat_rad) * (al**2 / 2 +
                                                         (5 - t + 9 * c + 4 * c**2) * al**4 / 24 +
                                                         (61 - 58 * t + t**2 + 600 * c - 330 * e_prime2) * al**6 / 720))
        if lat < 0:
            northing += 10000000.0  # False northing for southern hemisphere

        return easting, northing, zone_number, hemisphere

    @staticmethod
    def build_local_tangent_plane(gps_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Construct a Local East-North-Up (ENU) coordinate frame anchored at the centroid.
        """
        valid_coords = []
        for r in gps_records:
            lat = r.get("latitude") or r.get("lat")
            lon = r.get("longitude") or r.get("lon")
            alt = r.get("altitude") or r.get("alt") or 0.0
            if lat is not None and lon is not None:
                try:
                    lat_f = float(lat)
                    lon_f = float(lon)
                    alt_f = float(alt)
                    if -90.0 <= lat_f <= 90.0 and -180.0 <= lon_f <= 180.0:
                        easting, northing, zone, hemi = GPSService.latlon_to_utm(lat_f, lon_f)
                        valid_coords.append({
                            "lat": lat_f, "lon": lon_f, "alt": alt_f,
                            "easting": easting, "northing": northing,
                            "zone": zone, "hemisphere": hemi
                        })
                except (ValueError, TypeError):
                    continue

        if not valid_coords:
            return {
                "available": False,
                "reason": "No valid GPS coordinates found in telemetry data."
            }

        ref_lat = float(np.mean([c["lat"] for c in valid_coords]))
        ref_lon = float(np.mean([c["lon"] for c in valid_coords]))
        ref_alt = float(np.mean([c["alt"] for c in valid_coords]))
        ref_easting = float(np.mean([c["easting"] for c in valid_coords]))
        ref_northing = float(np.mean([c["northing"] for c in valid_coords]))
        zone = valid_coords[0]["zone"]
        hemi = valid_coords[0]["hemisphere"]

        # Local ENU positions
        local_positions = []
        for c in valid_coords:
            x_east = c["easting"] - ref_easting
            y_north = c["northing"] - ref_northing
            z_up = c["alt"] - ref_alt
            local_positions.append({"x": round(x_east, 3), "y": round(y_north, 3), "z": round(z_up, 3)})

        # Trajectory length
        total_dist = 0.0
        for i in range(len(valid_coords) - 1):
            total_dist += GPSService.haversine_distance(
                valid_coords[i]["lat"], valid_coords[i]["lon"],
                valid_coords[i+1]["lat"], valid_coords[i+1]["lon"]
            )

        return {
            "available": True,
            "coordinate_system": "WGS84_UTM_ENU",
            "utm_zone": f"{zone}{hemi}",
            "reference_anchor": {
                "latitude": ref_lat,
                "longitude": ref_lon,
                "altitude": ref_alt,
                "utm_easting": ref_easting,
                "utm_northing": ref_northing
            },
            "flight_trajectory_length_meters": round(total_dist, 2),
            "sampled_waypoints_count": len(valid_coords),
            "local_coordinates": local_positions
        }
