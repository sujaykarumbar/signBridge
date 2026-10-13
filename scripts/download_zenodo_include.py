"""
SignBridge - Zenodo INCLUDE Dataset Downloader & Processor
===========================================================
Source: https://zenodo.org/records/4010759
Title: INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition
Authors: AI4Bharat, IIT Madras (ACM Multimedia 2020)

This script automates:
1. Querying the Zenodo API for record 4010759.
2. Listing categories and files.
3. Downloading specific or all categories with progress bars.
4. Extracting the videos to datasets/include/raw_videos/
5. Extracting 134-dim MediaPipe (Pose + Hands) landmark sequences (.npy) for model training.
"""

import os
import sys
import json
import zipfile
import argparse
import requests
from pathlib import Path
from tqdm import tqdm

ZENODO_RECORD_ID = "4010759"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

BASE_DIR = Path(__file__).resolve().parents[1]
DATASETS_DIR = BASE_DIR / "datasets" / "include"
RAW_VIDEOS_DIR = DATASETS_DIR / "raw_videos"
LANDMARKS_DIR = DATASETS_DIR / "extracted_landmarks"


def get_zenodo_files():
    """Fetch file metadata from Zenodo API."""
    print(f"[*] Querying Zenodo API for record {ZENODO_RECORD_ID} ...")
    resp = requests.get(ZENODO_API_URL, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    files = data.get("files", [])
    return files


def list_files(files):
    """Display all available categories and zip files."""
    print(f"\n{'='*75}")
    print(f" Zenodo Record 4010759: INCLUDE Dataset Files ({len(files)} items)")
    print(f"{'='*75}")
    
    categories = {}
    for f in files:
        name = f["key"]
        size_mb = f["size"] / (1024 * 1024)
        cat = name.split("_")[0] if "_" in name else "Metadata"
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((name, size_mb, f["links"]["self"]))

    total_size_mb = sum(f["size"] / (1024 * 1024) for f in files)
    print(f"Total dataset size across all zip files: {total_size_mb / 1024:.2f} GB\n")

    for cat, items in sorted(categories.items()):
        cat_size = sum(sz for _, sz, _ in items)
        print(f"  [Category] {cat} ({len(items)} files, {cat_size:.1f} MB total)")
        for name, sz, _ in items:
            print(f"     - {name} ({sz:.1f} MB)")
    print(f"{'='*75}\n")


def download_file(url: str, dest_path: Path):
    """Download file with streaming progress bar."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        print(f"[+] Already downloaded: {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.1f} MB)")
        return True

    print(f"[-] Downloading {dest_path.name} ...")
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        with open(dest_path, "wb") as f, tqdm(
            desc=dest_path.name,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    return True


def extract_zip(zip_path: Path, extract_dir: Path):
    """Extract zip archive."""
    print(f"[*] Extracting {zip_path.name} to {extract_dir} ...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    print(f"[+] Extracted: {zip_path.name}")


def extract_landmarks_from_video(video_path: Path, output_path: Path):
    """Extract 134-dim MediaPipe (Pose + Hands) landmarks from a video file."""
    try:
        import cv2
        import numpy as np
        import mediapipe as mp

        mp_pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
        mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

        cap = cv2.VideoCapture(str(video_path))
        features_list = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            p_res = mp_pose.process(rgb)
            h_res = mp_hands.process(rgb)

            feats = np.zeros(134, dtype=np.float32)
            if p_res and p_res.pose_landmarks:
                for i, lm in enumerate(p_res.pose_landmarks.landmark[:25]):
                    feats[i * 2] = lm.x
                    feats[i * 2 + 1] = lm.y

            if h_res and h_res.multi_hand_landmarks:
                for h_idx, hand in enumerate(h_res.multi_hand_landmarks[:2]):
                    base = 50 + (h_idx * 42)
                    for i, lm in enumerate(hand.landmark[:21]):
                        feats[base + i * 2] = lm.x
                        feats[base + i * 2 + 1] = lm.y

            features_list.append(feats)

        cap.release()
        mp_pose.close()
        mp_hands.close()

        if features_list:
            seq_arr = np.array(features_list, dtype=np.float32)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(output_path), seq_arr)
            return True
    except Exception as e:
        print(f"[!] Error processing {video_path.name}: {e}")
    return False


def main():
    parser = argparse.ArgumentParser(description="Zenodo INCLUDE Dataset Downloader & Landmark Extractor")
    parser.add_argument("--list", action="store_true", help="List all categories and files on Zenodo")
    parser.add_argument("--category", type=str, help="Download a specific category (e.g. Greetings, Days_and_Time, People)")
    parser.add_argument("--extract", action="store_true", help="Extract downloaded zip files")
    parser.add_argument("--extract-landmarks", action="store_true", help="Extract MediaPipe 134-dim landmarks from extracted videos")
    args = parser.parse_args()

    files = get_zenodo_files()

    if args.list or len(sys.argv) == 1:
        list_files(files)
        if len(sys.argv) == 1:
            print("Tip: Use --category <name> to download a category (e.g. python scripts/download_zenodo_include.py --category Greetings)")
        return

    selected_files = []
    if args.category:
        query = args.category.lower()
        for f in files:
            if query in f["key"].lower():
                selected_files.append(f)

    if not selected_files:
        print(f"[!] No files matched category: {args.category}")
        return

    print(f"\n[+] Found {len(selected_files)} files matching '{args.category}':")
    for f in selected_files:
        print(f"   - {f['key']} ({f['size'] / (1024*1024):.1f} MB)")

    downloads_dir = DATASETS_DIR / "zips"
    downloads_dir.mkdir(parents=True, exist_ok=True)

    for f in selected_files:
        name = f["key"]
        dest = downloads_dir / name
        download_file(f["links"]["self"], dest)
        if args.extract or True:  # Extract automatically
            cat_name = name.split("_")[0]
            extract_zip(dest, RAW_VIDEOS_DIR / cat_name)

    if args.extract_landmarks:
        print("\n[*] Extracting MediaPipe landmarks from videos...")
        video_files = list(RAW_VIDEOS_DIR.rglob("*.MOV")) + list(RAW_VIDEOS_DIR.rglob("*.mp4"))
        print(f"[+] Found {len(video_files)} video clips to process.")
        for v in tqdm(video_files, desc="Processing videos"):
            rel = v.relative_to(RAW_VIDEOS_DIR)
            out_path = LANDMARKS_DIR / rel.with_suffix(".npy")
            if not out_path.exists():
                extract_landmarks_from_video(v, out_path)
        print(f"[OK] Landmarks extracted into {LANDMARKS_DIR}")


if __name__ == "__main__":
    main()
