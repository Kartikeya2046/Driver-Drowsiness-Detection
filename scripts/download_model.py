"""Download the MediaPipe FaceLandmarker model into models/.

The model file isn't committed (small but binary, easy to refetch).
"""
import urllib.request
from pathlib import Path

URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
OUT = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL} -> {OUT}")
    urllib.request.urlretrieve(URL, OUT)
    print(f"Done ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
