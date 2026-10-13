import os
import json
import requests
from pathlib import Path
from tqdm import tqdm

DATASETS_INCLUDE_DIR = Path("datasets/include")
SPLITS_DIR = DATASETS_INCLUDE_DIR / "splits"
MODELS_DIR = Path("backend/models")

DATASETS_INCLUDE_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/AI4Bharat/INCLUDE/master"

# 1. Download Label Maps
print("--- 1. Downloading AI4Bharat/INCLUDE Label Maps ---")
for fname in ["label_map_include50.json", "label_map_include.json"]:
    url = f"{GITHUB_RAW_BASE}/label_maps/{fname}"
    r = requests.get(url)
    if r.status_code == 200:
        (DATASETS_INCLUDE_DIR / fname).write_text(r.text, encoding="utf-8")
        (MODELS_DIR / fname).write_text(r.text, encoding="utf-8")
        print(f"Downloaded {fname} ({len(json.loads(r.text))} classes)")
    else:
        print(f"Failed to download {fname}: HTTP {r.status_code}")

# 2. Download Train / Val / Test Splits
print("\n--- 2. Downloading AI4Bharat/INCLUDE Splits ---")
for split_file in ["include50_train.txt", "include50_val.txt", "include50_test.txt"]:
    url = f"{GITHUB_RAW_BASE}/train_test_paths/{split_file}"
    r = requests.get(url)
    if r.status_code == 200:
        (SPLITS_DIR / split_file).write_text(r.text, encoding="utf-8")
        lines = len(r.text.strip().split("\n"))
        print(f"Downloaded {split_file} ({lines} video paths)")
    else:
        print(f"Failed to download {split_file}: HTTP {r.status_code}")

# 3. Download Pretrained AI4Bharat Weights (INCLUDE-50 No-CNN LSTM)
print("\n--- 3. Downloading AI4Bharat Pretrained LSTM Model (81.7 MB) ---")
target_model_path = MODELS_DIR / "include50_no_cnn_lstm.pth"
if not target_model_path.exists() or target_model_path.stat().st_size < 1000000:
    weights_url = "https://api.wandb.ai/files/abdur-ai4bharat/include50-no-cnn/1isx5nl6/augs_lstm.pth"
    print(f"Streaming from {weights_url} ...")
    r = requests.get(weights_url, stream=True)
    if r.status_code == 200:
        with open(target_model_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        print(f"Saved to {target_model_path} ({target_model_path.stat().st_size / (1024*1024):.2f} MB)")
    else:
        print(f"Failed to download weights: HTTP {r.status_code}")
else:
    print(f"Pretrained model already exists: {target_model_path} ({target_model_path.stat().st_size / (1024*1024):.2f} MB)")

print("\n--- Setup Complete ---")
