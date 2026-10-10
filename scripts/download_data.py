from pathlib import Path
import argparse
import hashlib
import json

import gdown


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "docs" / "datasets.json").read_text())
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=manifest)
    args = parser.parse_args()
    directory = root / "data"
    directory.mkdir(exist_ok=True)
    for item in manifest[args.dataset]:
        destination = directory / item["filename"]
        if destination.exists():
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            if digest == item["sha256"]:
                print(f"Already present: {destination.name}")
                continue
        temporary = destination.with_suffix(destination.suffix + ".part")
        result = gdown.download(id=item["drive_id"], output=str(temporary), quiet=False)
        if result is None:
            raise RuntimeError(f"Download failed: {item['filename']}")
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise ValueError(f"Checksum mismatch for {item['filename']}; the course data may have changed")
        temporary.replace(destination)
        print(f"Verified: {destination.name}")


if __name__ == "__main__":
    main()
