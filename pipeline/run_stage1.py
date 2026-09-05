"""
Stage 1 Runner: Video Ingestion and Intelligent Frame Extraction.
"""
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.video_processing import VideoProcessor
from pipeline.frame_extraction import FrameExtractor

def run_stage_1(video_path: str, fps_sample_rate: float = 2.0, max_frames: int = 120):
    print(f"\n==========================================")
    print(f"   ELEVATEX STAGE 1: VIDEO INGESTION")
    print(f"==========================================")
    print(f"Target Video: {video_path}")

    processor = VideoProcessor(video_path)
    res = processor.validate_and_inspect()
    if not res["valid"]:
        print(f"[ERROR] Video Validation Failed: {res.get('error')}")
        return False, res

    meta = res["metadata"]
    print(f"[OK] Resolution: {meta['resolution']} | FPS: {meta['fps']} | Duration: {meta['duration_formatted']} | Frames: {meta['total_frames']}")

    extractor = FrameExtractor()
    extract_res = extractor.extract_frames(video_path, fps_sample_rate=fps_sample_rate, max_frames=max_frames)
    print(f"[OK] Extracted {extract_res['extracted_frames_count']} frames into: {extract_res['output_directory']}")

    return True, {"video_meta": meta, "extraction_meta": extract_res}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ElevateX Stage 1: Video Ingestion & Extraction")
    parser.add_argument("--video", type=str, required=True, help="Path to input drone video")
    parser.add_argument("--fps", type=float, default=2.0, help="Frame sampling rate (FPS)")
    parser.add_argument("--max_frames", type=int, default=120, help="Max frames to extract")
    args = parser.parse_args()

    success, _ = run_stage_1(args.video, args.fps, args.max_frames)
    sys.exit(0 if success else 1)
