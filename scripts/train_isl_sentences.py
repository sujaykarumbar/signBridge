"""
SignBridge - Train 97 Continuous ISL Sentence Classifier
=========================================================
Trains a real-time continuous Indian Sign Language Sentence Classifier
directly on the 97 continuous sentences (663 signer video sequences)
in `datasets/isl-csltr/Frames_Sentence_Level`.

Features per sequence (Spatio-Temporal Trajectory):
  - 126 hand coordinates (wrist-centered) across sampled temporal keyframes
  - 24 pose coordinates (shoulders, elbows, wrists, nose)
  - Temporal pooling: Mean, Std, Min, Max, Delta (end - start), Velocity magnitude
  - Total feature vector: 900 dimensions per continuous sentence sequence

Output:
  - backend/models/isl_sentence_model.joblib (< 3ms inference on CPU)
  - backend/models/isl_sentence_classes.json
"""

import os
import sys
import json
import time
import glob
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import cv2
import joblib
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "datasets" / "isl-csltr" / "Frames_Sentence_Level"
MODELS_DIR = BASE_DIR / "backend" / "models"
OUTPUT_MODEL_PATH = MODELS_DIR / "isl_sentence_model.joblib"
OUTPUT_CLASSES_PATH = MODELS_DIR / "isl_sentence_classes.json"
CACHE_PATH = BASE_DIR / "datasets" / "isl-csltr" / "extracted_sentence_landmarks.npz"

NUM_FRAME_FEATS = 150  # 126 hand + 24 pose
NUM_TOTAL_FEATS = 900  # 6 statistical/temporal aggregations * 150


def clean_sentence_label(raw_name: str) -> str:
    """Format raw folder name into a natural English sentence."""
    name = raw_name.strip()
    if name.lower() == "i am (age)":
        return "I am fine."
    if name.lower() == "my name is xxxxxxxx":
        return "My name is John."
    # Capitalize first letter, ensure period/question mark
    name = name[0].upper() + name[1:] if len(name) > 1 else name.upper()
    if any(name.lower().startswith(q) for q in ["what", "where", "who", "why", "how", "when", "which", "shall", "can", "could", "do", "are", "had"]):
        if not name.endswith("?"):
            name = name.rstrip(".") + "?"
    elif not name.endswith((".", "!", "?")):
        name = name + "."
    return name


def _process_single_sequence(task):
    """Worker function for process pool to extract features from one sequence."""
    seq_dir, class_name = task
    frame_files = sorted(glob.glob(os.path.join(seq_dir, "*.jpg")))
    if not frame_files or len(frame_files) < 3:
        return None

    import mediapipe as mp
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.25,
        min_tracking_confidence=0.25
    )
    mp_pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.25,
        min_tracking_confidence=0.25
    )

    # Sample up to 12 frames across sequence
    n = len(frame_files)
    sample_count = min(12, n)
    indices = np.linspace(0, n - 1, sample_count, dtype=int)
    sampled_paths = [frame_files[i] for i in indices]

    frame_feats = []
    for fp in sampled_paths:
        img = cv2.imread(fp)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        hand_feat = np.zeros(126, dtype=np.float32)
        h_res = mp_hands.process(rgb)
        if h_res and h_res.multi_hand_landmarks:
            for h_idx, hand in enumerate(h_res.multi_hand_landmarks[:2]):
                base = h_idx * 63
                w = hand.landmark[0]
                for i, lm in enumerate(hand.landmark[:21]):
                    hand_feat[base + i * 3]     = lm.x - w.x
                    hand_feat[base + i * 3 + 1] = lm.y - w.y
                    hand_feat[base + i * 3 + 2] = lm.z - w.z

        pose_feat = np.zeros(24, dtype=np.float32)
        p_res = mp_pose.process(rgb)
        if p_res and p_res.pose_landmarks:
            ref = p_res.pose_landmarks.landmark[0]  # nose
            pose_indices = [0, 11, 12, 13, 14, 15, 16, 23]
            for p_i, idx in enumerate(pose_indices):
                lm = p_res.pose_landmarks.landmark[idx]
                pose_feat[p_i * 3]     = lm.x - ref.x
                pose_feat[p_i * 3 + 1] = lm.y - ref.y
                pose_feat[p_i * 3 + 2] = lm.z - ref.z

        feats_150 = np.concatenate([hand_feat, pose_feat])
        frame_feats.append(feats_150)

    mp_hands.close()
    mp_pose.close()

    if len(frame_feats) < 3:
        return None

    arr = np.array(frame_feats, dtype=np.float32)  # shape (T, 150)
    if np.count_nonzero(arr) < 15:
        return None

    mean_f = np.mean(arr, axis=0)
    std_f = np.std(arr, axis=0)
    min_f = np.min(arr, axis=0)
    max_f = np.max(arr, axis=0)
    delta_f = arr[-1] - arr[0]
    vel_f = np.mean(np.abs(np.diff(arr, axis=0)), axis=0) if len(arr) > 1 else np.zeros(150, dtype=np.float32)

    seq_vector = np.concatenate([mean_f, std_f, min_f, max_f, delta_f, vel_f]).astype(np.float32)
    return seq_vector, class_name


def main():
    if not DATA_DIR.exists():
        print(f"[!] Error: {DATA_DIR} does not exist.")
        sys.exit(1)

    if CACHE_PATH.exists():
        print(f"[*] Loading cached sentence landmarks from {CACHE_PATH}...")
        data = np.load(CACHE_PATH, allow_pickle=True)
        X = data["X"]
        y = data["y"]
        print(f"[OK] Loaded {len(X)} sentence samples across {len(set(y))} classes.")
    else:
        sentence_dirs = sorted([d for d in os.listdir(DATA_DIR) if (DATA_DIR / d).is_dir()])
        print(f"[*] Found {len(sentence_dirs)} continuous sentence classes in {DATA_DIR}")

        tasks = []
        for s_dir in sentence_dirs:
            signer_dirs = sorted(glob.glob(str(DATA_DIR / s_dir / "*")))
            for signer in signer_dirs:
                if os.path.isdir(signer):
                    clean_lbl = clean_sentence_label(s_dir)
                    tasks.append((signer, clean_lbl))

        print(f"[*] Extracting landmarks from {len(tasks)} sentence sequences using multi-core processing...")
        t0 = time.time()
        X = []
        y = []

        workers = min(6, os.cpu_count() or 4)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_process_single_sequence, t) for t in tasks]
            completed = 0
            for f in as_completed(futures):
                res = f.result()
                completed += 1
                if res is not None:
                    feat, label = res
                    X.append(feat)
                    y.append(label)
                if completed % 50 == 0 or completed == len(tasks):
                    print(f"   [{completed}/{len(tasks)}] extracted ({len(X)} valid samples so far)...")

        dt = time.time() - t0
        print(f"[OK] Extraction complete in {dt:.1f}s. Valid samples: {len(X)} across {len(set(y))} classes.")

        if len(X) < 50:
            print("[!] Not enough samples extracted.")
            sys.exit(1)

        X = np.array(X, dtype=np.float32)
        y = np.array(y)
        np.savez_compressed(CACHE_PATH, X=X, y=y)
        print(f"[OK] Saved extracted landmarks cache to {CACHE_PATH}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    print(f"\n[*] Training Continuous ISL Sentence Classifier ({len(set(y))} classes)...")
    clf = ExtraTreesClassifier(
        n_estimators=150,
        max_depth=25,
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
    validation_clf = ExtraTreesClassifier(
        n_estimators=150,
        max_depth=25,
        random_state=42,
        class_weight="balanced"
    )
    validation_clf.fit(X_train, y_train)
    validation_acc = float(accuracy_score(y_validation, validation_clf.predict(X_validation)))
    print(f"[OK] Held-out sequence accuracy: {validation_acc * 100:.2f}%")

    clf.fit(X, y)

    train_acc = float(accuracy_score(y, clf.predict(X)))
    print(f"[OK] Model Training Accuracy: {train_acc * 100:.2f}%")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    classes_list = sorted(list(clf.classes_))
    joblib.dump({
        "model": clf,
        "classes": classes_list,
        "num_features": NUM_TOTAL_FEATS,
        "accuracy": train_acc,
        "validation_accuracy": validation_acc,
    }, OUTPUT_MODEL_PATH)

    with open(OUTPUT_CLASSES_PATH, "w", encoding="utf-8") as f:
        json.dump(classes_list, f, indent=2)

    print(f"[SUCCESS] Model saved to {OUTPUT_MODEL_PATH}")
    print(f"[SUCCESS] Classes list saved to {OUTPUT_CLASSES_PATH}")


if __name__ == "__main__":
    main()
