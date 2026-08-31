"""Download IEEE-CIS competition data and place it in data/.

Requires:
  1. ~/.kaggle/kaggle.json with a valid API token
  2. Competition rules accepted at
     https://www.kaggle.com/competitions/ieee-fraud-detection/rules
"""
import shutil
import sys
from pathlib import Path

import kagglehub

import config

WANTED = ["train_transaction.csv", "train_identity.csv"]


def main() -> int:
    try:
        src = Path(kagglehub.competition_download("ieee-fraud-detection"))
    except Exception as exc:  # noqa: BLE001 - surface the real cause plainly
        if "403" in str(exc):
            print(
                "403 from Kaggle.\n\n"
                "Accept the competition rules first:\n"
                "  https://www.kaggle.com/competitions/ieee-fraud-detection/rules\n"
                "Then re-run this script.",
                file=sys.stderr,
            )
            return 1
        raise

    print(f"downloaded to {src}")
    config.DATA_DIR.mkdir(exist_ok=True)

    for name in WANTED:
        matches = list(src.rglob(name))
        if not matches:
            print(f"  !! {name} not found in download", file=sys.stderr)
            continue
        dest = config.DATA_DIR / name
        if dest.exists():
            print(f"  = {name} already in data/, skipping")
            continue
        shutil.copy2(matches[0], dest)
        mb = dest.stat().st_size / 1e6
        print(f"  + {name}  ({mb:,.0f} MB)")

    print(f"\nready. next:  python data.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
