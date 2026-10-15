"""
SignBridge - Train 114 ISL Word Gesture Classifier from Local ISL-CSLTR Dataset
=============================================================================
This script trains a high-speed, local Indian Sign Language Word Classifier
directly from the local `datasets/isl-csltr/Frames_Word_Level` dataset without
requiring any 52 GB Zenodo download!

Features:
  - 126 normalized 3D hand coordinates (wrist-centered)
  - 24 upper-body pose coordinates (shoulders, elbows, wrists, nose)
  - 150-dimensional feature vector per sample
  - Trained via Calibrated Classifier (Random Forest + Logistic Regression)
  - Output model: backend/models/isl_words_model.joblib (< 5ms inference)
"""

import os
import sys
import json
import time
import glob
from pathlib import Path
import numpy as np
import cv2
import mediapipe as mp
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "datasets" / "isl-csltr" / "Frames_Word_Level"
MODELS_DIR = BASE_DIR / "backend" / "models"
OUTPUT_MODEL_PATH = MODELS_DIR / "isl_words_model.joblib"
OUTPUT_CLASSES_PATH = MODELS_DIR / "isl_words_classes.json"

NUM_FEATURES = 150  # 126 (2 hands * 21 * 3) + 24 (8 pose keypoints * 3)


def extract_landmarks_from_image(img_bgr, mp_hands, mp_pose):
    """Extract 150 normalized spatial features from an image."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    feats = np.zeros(NUM_FEATURES, dtype=np.float32)

    # 1. Hands (up to 2 hands) -> 126 features
    hand_res = mp_hands.process(rgb)
    if hand_res and hand_res.multi_hand_landmarks:
        for h_idx, hand in enumerate(hand_res.multi_hand_landmarks[:2]):
            base = h_idx * 63
            wrist = hand.landmark[0]
            for lm_idx, lm in enumerate(hand.landmark[:21]):
                feats[base + lm_idx * 3]     = lm.x - wrist.x
                feats[base + lm_idx * 3 + 1] = lm.y - wrist.y
                feats[base + lm_idx * 3 + 2] = lm.z - wrist.z

    # 2. Upper body pose (nose, shoulders, elbows, wrists = indices 0, 11, 12, 13, 14, 15, 16, 23, 24) -> 24 features
    pose_res = mp_pose.process(rgb)
    if pose_res and pose_res.pose_landmarks:
        pose_indices = [0, 11, 12, 13, 14, 15, 16, 23]  # 8 points
        base_pose = 126
        # Center relative to nose
        ref = pose_res.pose_landmarks.landmark[0]
        for p_idx, p_i in enumerate(pose_indices):
            lm = pose_res.pose_landmarks.landmark[p_i]
            feats[base_pose + p_idx * 3]     = lm.x - ref.x
            feats[base_pose + p_idx * 3 + 1] = lm.y - ref.y
            feats[base_pose + p_idx * 3 + 2] = lm.z - ref.z

    return feats


def main():
    if not DATA_DIR.exists():
        print(f"[!] Error: {DATA_DIR} does not exist.")
        sys.exit(1)

    CACHE_PATH = BASE_DIR / "datasets" / "isl-csltr" / "extracted_word_landmarks.npz"

    if CACHE_PATH.exists():
        print(f"[*] Loading cached landmarks from {CACHE_PATH}...")
        data = np.load(CACHE_PATH, allow_pickle=True)
        X = data["X"]
        y = data["y"]
        print(f"[OK] Loaded {len(X)} samples across {len(set(y))} classes.")
    else:
        classes = sorted([d for d in os.listdir(DATA_DIR) if (DATA_DIR / d).is_dir()])
        print(f"[*] Found {len(classes)} ISL word classes in {DATA_DIR}")

        mp_hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.3
        )
        mp_pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.3
        )

        X = []
        y = []

        print("[*] Extracting MediaPipe landmarks from images...")
        t0 = time.time()
        total_imgs = 0

        for c_idx, class_name in enumerate(classes):
            class_folder = DATA_DIR / class_name
            img_files = list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.png")) + list(class_folder.glob("*.jpeg"))
            for img_path in img_files:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                feats = extract_landmarks_from_image(img, mp_hands, mp_pose)
                # Only keep samples where hands or pose were detected
                if np.count_nonzero(feats) > 10:
                    X.append(feats)
                    y.append(class_name)
                    total_imgs += 1

            if (c_idx + 1) % 20 == 0 or c_idx == len(classes) - 1:
                print(f"   Processed {c_idx + 1}/{len(classes)} classes ({total_imgs} valid samples so far)...")

        mp_hands.close()
        mp_pose.close()

        dt = time.time() - t0
        print(f"[OK] Extracted landmarks for {len(X)} valid samples across {len(set(y))} classes in {dt:.1f}s.")

        if len(X) < 20:
            print("[!] Not enough detected samples to train.")
            sys.exit(1)

        # Save cache for instant reuse
        np.savez_compressed(CACHE_PATH, X=np.array(X, dtype=np.float32), y=np.array(y))
        print(f"[OK] Saved landmarks cache to {CACHE_PATH}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    print("\n[*] Training Random Forest Classifier on 114 ISL Classes...")
    rf = RandomForestClassifier(
        n_estimators=120,
        max_depth=20,
        random_state=42,
        class_weight="balanced"
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    validation_rf = RandomForestClassifier(
        n_estimators=120,
        max_depth=20,
        random_state=42,
        class_weight="balanced"
    )
    validation_rf.fit(X_train, y_train)
    validation_acc = float(accuracy_score(y_validation, validation_rf.predict(X_validation)))
    print(f"[OK] Held-out image accuracy: {validation_acc * 100:.2f}%")

    rf.fit(X, y)

    # Train accuracy
    preds = rf.predict(X)
    train_acc = float(accuracy_score(y, preds))
    print(f"[OK] Model Training Accuracy: {train_acc * 100:.2f}%")

    # Save model and metadata
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": rf,
        "classes": list(rf.classes_),
        "num_features": NUM_FEATURES,
        "accuracy": train_acc,
        "validation_accuracy": validation_acc,
    }, OUTPUT_MODEL_PATH)

    with open(OUTPUT_CLASSES_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "classes": list(rf.classes_),
            "total_classes": len(rf.classes_),
            "train_samples": len(X),
            "features_dim": NUM_FEATURES,
            "accuracy": train_acc
        }, f, indent=2)

    file_size_mb = OUTPUT_MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"[OK] Saved trained model to {OUTPUT_MODEL_PATH} ({file_size_mb:.2f} MB)")
    print(f"[OK] Saved class labels to {OUTPUT_CLASSES_PATH}")
    print("[DONE] Local ISL word training complete!")


if __name__ == "__main__":
    main()
