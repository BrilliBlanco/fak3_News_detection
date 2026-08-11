"""
One-command dataset setup.

The repo ships `archive.zip` (the Kaggle "Fake and Real News Dataset").
This extracts Fake.csv / True.csv out of it into `data/`, wherever they
happen to sit inside the zip, and verifies them so nobody has to follow
five README steps by hand.

Usage:
    python src/setup_data.py
    python src/setup_data.py --archive some_other.zip --force
"""

import argparse
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

from config import DATA_DIR, PROJECT_ROOT

WANTED = ("Fake.csv", "True.csv")


def parse_args():
    p = argparse.ArgumentParser(description="Extract the news dataset into data/.")
    p.add_argument("--archive", default=str(PROJECT_ROOT / "archive.zip"),
                   help="Zip file containing Fake.csv and True.csv")
    p.add_argument("--data-dir", default=str(DATA_DIR), help="Where to put the CSVs")
    p.add_argument("--force", action="store_true", help="Overwrite CSVs that already exist")
    return p.parse_args()


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main():
    args = parse_args()
    archive = Path(args.archive)
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    existing = [n for n in WANTED if (data_dir / n).exists()]
    if existing and not args.force:
        print(f"Already present in {data_dir}: {', '.join(existing)}")
        print("Nothing to do (use --force to re-extract).")
        return 0

    if not archive.exists():
        print(f"Archive not found: {archive}", file=sys.stderr)
        print("Download it from Kaggle and drop Fake.csv / True.csv into data/ manually:",
              file=sys.stderr)
        print("https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset",
              file=sys.stderr)
        return 1

    with zipfile.ZipFile(archive) as zf:
        # Match on basename so it works whether or not the zip has a top folder
        members = {Path(i.filename).name: i for i in zf.infolist() if not i.is_dir()}
        missing = [n for n in WANTED if n not in members]
        if missing:
            print(f"{archive.name} does not contain {', '.join(missing)}.", file=sys.stderr)
            print(f"It contains: {sorted(members)}", file=sys.stderr)
            return 1

        for name in WANTED:
            info = members[name]
            target = data_dir / name
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            print(f"  extracted {name:<10} {target.stat().st_size / 1e6:>7.1f} MB  "
                  f"sha256:{sha256(target)[:12]}")

    print(f"\nDataset ready in {data_dir.resolve()}")
    print("Next: python src/eda.py     (explore it)")
    print("      python src/train.py   (train the models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
