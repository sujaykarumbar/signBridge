# SignBridge Datasets (Indian Sign Language - ISL)

This folder contains Indian Sign Language (ISL) datasets downloaded and integrated for SignBridge.

---

## 1. ISL Alphabets & Digits (`indian-sign-language-isl`) — ✅ ADDED
- **Source**: [prathumarikeri/indian-sign-language-isl](https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl)
- **Status**: **Downloaded & Extracted** in [`datasets/indian-sign-language-isl/`](file:///c:/Users/USER/Downloads/signBridge/signBridge/datasets/indian-sign-language-isl)
- **Total Images**: **42,745 JPG images**
- **Classes (35 total)**:
  - **Digits (9)**: `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`
  - **Alphabets (26)**: `A`, `B`, `C`, `D`, `E`, `F`, `G`, `H`, `I`, `J`, `K`, `L`, `M`, `N`, `O`, `P`, `Q`, `R`, `S`, `T`, `U`, `V`, `W`, `X`, `Y`, `Z`
- **Trained Model**: [`backend/models/isl_alphabet_model.joblib`](file:///c:/Users/USER/Downloads/signBridge/signBridge/backend/models/isl_alphabet_model.joblib) (Two-handed MediaPipe landmark classifier, **99.75% validation accuracy**).

---

## 2. ISL-CSLTR Continuous Sentences (`isl-csltr`) — ✅ DOWNLOADED (8.9 GB)
- **Source**: [drblack00/isl-csltr-indian-sign-language-dataset](https://www.kaggle.com/datasets/drblack00/isl-csltr-indian-sign-language-dataset)
- **Status**: **100% Downloaded (8.29 GB)**, extracting into [`datasets/isl-csltr/`](file:///c:/Users/USER/Downloads/signBridge/signBridge/datasets)
- **Content**: 700 fully annotated videos across 100 continuous Indian Sign Language sentences.

---

## 3. ASL Signs (`asl-signs`) — ⚠️ EXCLUDED (American Sign Language)
- **Note**: This dataset is for **American Sign Language (ASL)**, which uses single-handed grammar and gestures that are completely different from **Indian Sign Language (ISL)**. Per user requirement (*"i want for the isl"*), this ASL dataset is excluded so SignBridge remains 100% focused on Indian Sign Language.

---

## 4. Current Dynamic Words Model (76 ISL Words) — ✅ ACTIVE ON PHONE
- **Model**: `Creator-090/isl-swin3d-model` on Hugging Face (Kinetics-400 Swin3D with real-time MediaPipe skeletal overlays).
- **Supported Signs (76 words)**:
  - `afternoon`, `animal`, `bad`, `beautiful`, `big`, `bird`, `blind`, `cat`, `cheap`, `clothing`, `cold`, `cow`, `curved`, `deaf`, `dog`, `dress`, `dry`, `evening`, `expensive`, `famous`, `fast`, `female`, `fish`, `flat`, `friday`, `good`, `happy`, `hat`, `healthy`, `horse`, `hot`, `hour`, `light`, `long`, `loose`, `loud`, `minute`, `monday`, `month`, `morning`, `mouse`, `narrow`, `new`, `night`, `old`, `pant`, `pocket`, `quiet`, `sad`, `saturday`, `second`, `shirt`, `shoes`, `short`, `sick`, `skirt`, `slow`, `small`, `suit`, `sunday`, `t_shirt`, `tall`, `thursday`, `time`, `today`, `tomorrow`, `tuesday`, `ugly`, `warm`, `wednesday`, `week`, `wet`, `wide`, `year`, `yesterday`, `young`

---

## 5. AI4Bharat/INCLUDE Dataset (`datasets/include`) — ✅ FULLY INTEGRATED & CONNECTED
- **Source**: [Zenodo Record 4010759](https://zenodo.org/records/4010759) & [AI4Bharat/INCLUDE (IIT Madras)](https://github.com/AI4Bharat/INCLUDE)
- **Publication**: *INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition* (ACM Multimedia 2020)
- **Dataset Structure (Zenodo)**:
  - 4,292 Video clips (52.85 GB) across 15 semantic categories:
    *Adjectives, Animals, Clothes, Colours, Days and Time, Electronics, Greetings, Home, Jobs, Means of Transportation, People, Places, Pronouns, Seasons, Society*
  - **Full Vocabulary**: 263 Indian Sign Language classes
  - **Core Benchmark**: 50 evaluation classes
- **Downloader & Landmark Extractor Tool**:
  - [`scripts/download_zenodo_include.py`](file:///c:/Users/USER/Downloads/signBridge/signBridge/scripts/download_zenodo_include.py):
    ```bash
    # List all 46 archives and categories from Zenodo API
    python scripts/download_zenodo_include.py --list

    # Download and extract specific category (e.g. Greetings)
    python scripts/download_zenodo_include.py --category Greetings

    # Extract 134-dim MediaPipe landmark sequences (.npy) for model training
    python scripts/download_zenodo_include.py --category Greetings --extract-landmarks
    ```
- **Backend API Endpoints**:
  - `GET /dataset/include/info` - Metadata, authors, Zenodo DOI, and stats.
  - `GET /dataset/include/categories` - List of all 15 semantic categories.
  - `GET /dataset/include/signs?category=...&query=...&core_only=true` - Filterable 263 ISL signs list.
- **Frontend Integration**:
  - **ISL Dictionary Screen** ([`include_vocabulary_screen.dart`](file:///c:/Users/USER/Downloads/signBridge/signBridge/frontend/lib/features/vocabulary/screens/include_vocabulary_screen.dart)):
    Searchable, category-filtered browser for all 263 signs with "Core-50" badges, dataset details, and a 1-tap "Practice with Camera" button opening live translation.
- **Pretrained Weights**:
  - [`backend/models/include50_no_cnn_lstm.pth`](file:///c:/Users/USER/Downloads/signBridge/signBridge/backend/models/include50_no_cnn_lstm.pth) (85.6 MB, official AI4Bharat BiLSTM landmark model).
- **Inference Service**:
  - [`backend/app/services/include_classifier_service.py`](file:///c:/Users/USER/Downloads/signBridge/signBridge/backend/app/services/include_classifier_service.py) (134-dimensional MediaPipe Hands + Pose landmark sequences).

## Gemini video translation evaluation

Use the labeled ISL-CSLTR sentence videos to measure Gemini output rather than trusting its self-reported confidence:

```powershell
python scripts\evaluate_gemini_isl.py --samples 10
python scripts\evaluate_gemini_isl.py --samples 10 --run
```

The first command only previews a deterministic, class-balanced sample. The second sends those public dataset clips through the same masked-frame upload pipeline used by the app and reports exact-match accuracy and failures. It can consume Gemini quota. It does not read personal videos or train a model. The service tries `gemini-3.5-flash` as a fallback after the configured model; set `GEMINI_FALLBACK_MODELS` in `backend/.env` to override that comma-separated list with models currently supported by the API key. Model availability does not guarantee translation accuracy.
