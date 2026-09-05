"""
Video Processing Module.
Validates input video file integrity and format compatibility.
"""
from pathlib import Path
from typing import Dict, Any
from services.metadata_service import MetadataService

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

class VideoProcessor:
    def __init__(self, video_path: str):
        self.video_path = Path(video_path)

    def validate_and_inspect(self) -> Dict[str, Any]:
        """Verify video exists, check extension, and extract intrinsic properties."""
        if not self.video_path.exists():
            return {
                "valid": False,
                "error": f"Video file not found at {self.video_path}"
            }

        ext = self.video_path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return {
                "valid": False,
                "error": f"Unsupported video format '{ext}'. Allowed formats: {', '.join(SUPPORTED_EXTENSIONS)}"
            }

        try:
            metadata = MetadataService.extract_video_metadata(str(self.video_path))
            if metadata["total_frames"] < 5:
                return {
                    "valid": False,
                    "error": f"Video contains insufficient frames ({metadata['total_frames']}). Minimum required is 5."
                }

            return {
                "valid": True,
                "metadata": metadata
            }
        except Exception as e:
            return {
                "valid": False,
                "error": f"Failed to read video stream: {str(e)}"
            }
