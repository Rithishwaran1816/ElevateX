"""
ElevateX Standalone 3D Viewer & Server Launcher.
"""
import sys
import webbrowser
from pathlib import Path
from app import run_server

def main():
    port = 5000
    url = f"http://127.0.0.1:{port}"
    print(f"Launching ElevateX 3D Viewer on {url}...")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    run_server(host="127.0.0.1", port=port, debug=False)

if __name__ == "__main__":
    main()
