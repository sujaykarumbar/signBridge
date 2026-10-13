import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class INCLUDEDatasetService:
    """
    Service for accessing and querying the official AI4Bharat INCLUDE Dataset
    from Zenodo Record 4010759 (https://zenodo.org/records/4010759).
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(INCLUDEDatasetService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        self.root_dir = Path(__file__).resolve().parents[2]
        self.include50_labels_path = self.root_dir / "models" / "label_map_include50.json"
        self.include_all_labels_path = self.root_dir / "models" / "label_map_include.json"
        self.splits_dir = self.root_dir.parents[0] / "datasets" / "include" / "splits"

        self.include50_classes = set()
        self.all_classes = {}
        self.category_map = {}
        self._load_dataset()
        self._initialized = True

    def _load_dataset(self):
        try:
            if self.include50_labels_path.exists():
                with open(self.include50_labels_path, "r", encoding="utf-8") as f:
                    self.include50_classes = set(json.load(f).keys())

            if self.include_all_labels_path.exists():
                with open(self.include_all_labels_path, "r", encoding="utf-8") as f:
                    self.all_classes = json.load(f)

            # Map from splits
            if self.splits_dir.exists():
                for split_file in self.splits_dir.glob("*.txt"):
                    with open(split_file, "r", encoding="utf-8") as f:
                        for line in f:
                            parts = line.strip().split("/")
                            if len(parts) >= 2:
                                cat = parts[0].replace("_", " ")
                                sign_raw = parts[1]
                                # e.g. "48. Hello" -> "hello"
                                sign_clean = sign_raw.split(". ")[-1].strip().lower().replace(" ", "").replace("-", "")
                                self.category_map[sign_clean] = {
                                    "category": cat,
                                    "display_name": sign_raw.split(". ")[-1].strip().title(),
                                }

        except Exception as e:
            logger.error("Failed to load INCLUDE dataset files: %s", e)

    def get_info(self) -> dict:
        return {
            "title": "INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition",
            "source": "Zenodo Record 4010759",
            "zenodo_url": "https://zenodo.org/records/4010759",
            "authors": "AI4Bharat, IIT Madras",
            "paper": "ACM Multimedia 2020",
            "paper_url": "https://dl.acm.org/doi/10.1145/3394171.3413528",
            "total_videos": 4292,
            "total_classes": len(self.all_classes) if self.all_classes else 263,
            "core_evaluation_classes": len(self.include50_classes) if self.include50_classes else 50,
            "categories_count": 15,
            "trained_model_weights": "include50_no_cnn_lstm.pth (85.6 MB)",
            "classes_50": sorted(list(self.include50_classes)),
        }

    def get_categories(self) -> list[str]:
        return [
            "All",
            "Greetings",
            "People",
            "Days and Time",
            "Adjectives",
            "Places",
            "Home",
            "Colours",
            "Animals",
            "Clothes",
            "Jobs",
            "Means of Transportation",
            "Electronics",
            "Pronouns",
            "Seasons",
            "Society",
        ]

    def get_signs(self, category: str = None, query: str = None, core_only: bool = False) -> list[dict]:
        results = []
        # Fallback category mapping for classes not found in 50 split
        cat_inference = {
            "greetings": ["hello", "goodmorning", "thankyou", "namaste", "welcome", "sorry"],
            "people": ["boy", "girl", "father", "mother", "brother", "sister", "baby", "adult", "actor", "artist", "author"],
            "days and time": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "time", "hour", "year", "today", "tomorrow", "yesterday", "afternoon"],
            "adjectives": ["good", "bad", "hot", "cold", "biglarge", "smalllittle", "happy", "sad", "dry", "wet", "short", "tall", "beautiful", "alive"],
            "animals": ["dog", "cow", "bird", "cat", "horse", "mouse", "fish", "animal"],
            "clothes": ["hat", "shoes", "tshirt", "t_shirt", "pant", "shirt", "skirt", "suit", "dress"],
            "places": ["court", "bank", "house", "storeorshop", "bedroom", "bathroom", "school"],
            "home": ["window", "paint", "pen", "bed", "book", "box", "bag", "ball"],
            "jobs": ["teacher", "priest"],
            "transport": ["car", "trainticket", "bicycle", "boat"],
            "electronics": ["cellphone", "fan"],
            "pronouns": ["i", "you", "it", "youplural"],
            "colours": ["black", "white", "red", "blue"],
            "society": ["election", "death", "attack", "bill"],
        }

        classes_to_iterate = self.include50_classes if core_only else (self.all_classes.keys() or self.include50_classes)

        for c in sorted(classes_to_iterate):
            c_clean = c.lower().replace("_", "").replace("-", "")
            meta = self.category_map.get(c_clean, {})
            cat = meta.get("category")

            if not cat:
                # Infer category
                for c_k, signs in cat_inference.items():
                    if any(s in c_clean for s in signs):
                        cat = c_k.title()
                        break
                if not cat:
                    cat = "General"

            display_name = meta.get("display_name", c.replace("_", " ").title())
            is_core = c in self.include50_classes

            if category and category.lower() != "all" and cat.lower() != category.lower():
                continue

            if query and (query.lower() not in display_name.lower() and query.lower() not in c.lower()):
                continue

            results.append({
                "id": c,
                "label": display_name,
                "category": cat,
                "is_core_50": is_core,
                "dataset": "INCLUDE (Zenodo 4010759)"
            })

        return results
