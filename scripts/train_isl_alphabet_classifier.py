import os
import glob
import time
import numpy as np
import cv2
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

DATASET_DIR = "datasets/indian-sign-language-isl"
OUTPUT_MODEL = "backend/models/isl_alphabet_model.joblib"
IMAGES_PER_CLASS = 60

print(f"Loading dataset from: {DATASET_DIR}")
classes = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
print(f"Found {len(classes)} classes: {classes}")

mp_hands = mp.solutions.hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.4
)

X = []
y = []

t0 = time.time()
print(f"Extracting MediaPipe hand landmarks (sampling {IMAGES_PER_CLASS} images/class)...")

for label in classes:
    class_folder = os.path.join(DATASET_DIR, label)
    all_imgs = glob.glob(os.path.join(class_folder, "*.jpg"))
    sample_imgs = all_imgs[:IMAGES_PER_CLASS]
    
    extracted_count = 0
    for img_path in sample_imgs:
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = mp_hands.process(rgb)
        
        if res.multi_hand_landmarks:
            # 2 hands max: 21 landmarks * 3 coords * 2 hands = 126 features
            features = np.zeros(126, dtype=np.float32)
            for h_idx, hand_landmarks in enumerate(res.multi_hand_landmarks[:2]):
                base_idx = h_idx * 63
                # Normalize relative to wrist (landmark 0)
                wrist = hand_landmarks.landmark[0]
                for lm_idx, lm in enumerate(hand_landmarks.landmark):
                    features[base_idx + lm_idx * 3]     = lm.x - wrist.x
                    features[base_idx + lm_idx * 3 + 1] = lm.y - wrist.y
                    features[base_idx + lm_idx * 3 + 2] = lm.z - wrist.z
            
            X.append(features)
            y.append(label)
            extracted_count += 1
            
    print(f"  Class '{label}': extracted {extracted_count}/{len(sample_imgs)} hands")

mp_hands.close()

X = np.array(X)
y = np.array(y)
print(f"\nTotal landmark samples: {len(X)} with feature shape: {X.shape}")
print(f"Feature extraction time: {time.time() - t0:.1f}s")

if len(X) < 100:
    print("Error: Too few samples extracted.")
    exit(1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training RandomForest on {len(X_train)} samples...")

clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Validation Accuracy: {acc * 100:.2f}%")

os.makedirs(os.path.dirname(OUTPUT_MODEL), exist_ok=True)
joblib.dump({"model": clf, "classes": classes}, OUTPUT_MODEL)
print(f"Model saved to: {OUTPUT_MODEL} ({os.path.getsize(OUTPUT_MODEL) / 1024:.1f} KB)")
