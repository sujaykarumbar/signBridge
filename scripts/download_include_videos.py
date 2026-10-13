"""
Download category video zips from AI4Bharat/INCLUDE Zenodo repository (Record 4010759).
Allows selective downloading of specific categories (e.g., Greetings, Days_and_Time, Pronouns).
"""

import sys
import argparse
import requests
from pathlib import Path
import zipfile

ZENODO_BASE = "https://zenodo.org/api/records/4010759/files"

CATEGORY_FILES = {
    "greetings": ["Greetings_1of2.zip", "Greetings_2of2.zip"],
    "days_and_time": ["Days_and_Time_1of3.zip", "Days_and_Time_2of3.zip", "Days_and_Time_3of3.zip"],
    "pronouns": ["Pronouns_1of2.zip", "Pronouns_2of2.zip"],
    "animals": ["Animals_1of2.zip", "Animals_2of2.zip"],
    "clothes": ["Clothes_1of2.zip", "Clothes_2of2.zip"],
    "colours": ["Colours_1of2.zip", "Colours_2of2.zip"],
    "home": ["Home_1of4.zip", "Home_2of4.zip", "Home_3of4.zip", "Home_4of4.zip"],
    "people": ["People_1of5.zip", "People_2of5.zip", "People_3of5.zip", "People_4of5.zip", "People_5of5.zip"],
    "places": ["Places_1of4.zip", "Places_2of4.zip", "Places_3of4.zip", "Places_4of4.zip"],
    "jobs": ["Jobs_1of2.zip", "Jobs_2of2.zip"],
}


def download_category(category: str, dest_dir: Path = Path("datasets/include/raw_videos")):
    cat_key = category.lower().replace(" ", "_")
    if cat_key not in CATEGORY_FILES:
        print(f"Unknown category '{category}'. Available: {list(CATEGORY_FILES.keys())}")
        return False

    dest_dir.mkdir(parents=True, exist_ok=True)
    files = CATEGORY_FILES[cat_key]

    for fname in files:
        target = dest_dir / fname
        if target.exists() and target.stat().st_size > 1000:
            print(f"Already downloaded: {fname} ({target.stat().st_size / (1024*1024):.1f} MB)")
            continue

        url = f"{ZENODO_BASE}/{fname}/content"
        print(f"Downloading {fname} from {url}...")
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with open(target, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            print(f"Successfully saved {fname} ({target.stat().st_size / (1024*1024):.1f} MB)")
        else:
            print(f"Failed to download {fname}: HTTP {r.status_code}")
            return False

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download AI4Bharat/INCLUDE category videos.")
    parser.add_argument(
        "--category",
        type=str,
        default="greetings",
        help=f"Category to download. Choices: {list(CATEGORY_FILES.keys())}"
    )
    args = parser.parse_args()
    download_category(args.category)
