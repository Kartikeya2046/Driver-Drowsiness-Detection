"""Download UTA-RLDD (face-cropped) dataset from Kaggle into data/raw/.

Requires Kaggle API credentials at ~/.kaggle/kaggle.json
(https://www.kaggle.com/docs/api -> "Create New Token").
"""
import zipfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

DATASET = "mathiasviborg/uta-rldd-videos-cropped-by-faces"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading {DATASET} -> {RAW_DIR}")
    api.dataset_download_files(DATASET, path=str(RAW_DIR), unzip=False)

    zips = list(RAW_DIR.glob("*.zip"))
    if not zips:
        raise SystemExit("No zip file downloaded — check dataset slug / credentials.")

    for zip_path in zips:
        print(f"Extracting {zip_path.name}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(RAW_DIR)
        zip_path.unlink()

    print("Done. Contents of data/raw:")
    for p in sorted(RAW_DIR.iterdir()):
        print(" ", p.name)


if __name__ == "__main__":
    main()
