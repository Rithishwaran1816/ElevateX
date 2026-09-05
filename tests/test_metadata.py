"""
Unit Tests for Metadata and GPS Services.
"""
import pytest
import json
from pathlib import Path
from services.metadata_service import MetadataService
from services.gps_service import GPSService

def test_latlon_to_utm_conversion():
    # Test coordinates (New Delhi: 28.6139° N, 77.2090° E)
    lat, lon = 28.6139, 77.2090
    easting, northing, zone, hemi = GPSService.latlon_to_utm(lat, lon)

    assert zone == 43, f"Expected UTM zone 43, got {zone}"
    assert hemi == 'N'
    assert 700000 < easting < 750000, f"Easting {easting} outside expected range"
    assert 3100000 < northing < 3200000, f"Northing {northing} outside expected range"

def test_haversine_distance():
    # Distance between points ~111km per degree latitude
    lat1, lon1 = 28.0, 77.0
    lat2, lon2 = 29.0, 77.0
    dist_m = GPSService.haversine_distance(lat1, lon1, lat2, lon2)

    assert 110000 < dist_m < 112000, f"1 degree lat should be ~111km, got {dist_m}m"

def test_sensor_status_summary():
    # When no telemetry is provided
    status_none = MetadataService.get_sensor_status_summary(None)
    assert status_none["GPS"] == "NOT AVAILABLE"
    assert status_none["Flight Metadata"] == "NOT AVAILABLE"

    # When telemetry has GPS and IMU
    sample_tel = {
        "status": "AVAILABLE",
        "summary": {
            "has_gps": True,
            "has_imu": True,
            "has_rtk": True,
            "has_camera_intrinsics": False
        }
    }
    status_avail = MetadataService.get_sensor_status_summary(sample_tel)
    assert status_avail["GPS"] == "AVAILABLE"
    assert status_avail["Flight Metadata"] == "AVAILABLE"
    assert status_avail["IMU"] == "AVAILABLE"
    assert status_avail["RTK"] == "AVAILABLE"
    assert status_avail["Camera Intrinsics"] == "NOT AVAILABLE"
