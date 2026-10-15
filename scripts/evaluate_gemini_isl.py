"""Evaluate Gemini translations against a small, balanced sample of labeled ISL videos.

The source clips come from the already-downloaded ISL-CSLTR corpus. Use --run to
explicitly send the selected public corpus clips to Gemini; personal videos are
never read by this script.
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
VIDEO_ROOT = ROOT / "datasets" / "isl-csltr" / "Videos_Sentence_Level"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def normalized_sentence(value: str) -> str:
    """Normalize case and punctuation for exact-match sentence scoring."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", value.casefold())).strip()


def choose_samples(sample_count: int, seed: int) -> list[tuple[str, Path]]:
    if not VIDEO_ROOT.is_dir():
        raise FileNotFoundError(f"Labeled ISL-CSLTR videos not found: {VIDEO_ROOT}")

    classes = []
    for class_dir in sorted(path for path in VIDEO_ROOT.iterdir() if path.is_dir()):
        clips = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        )
        if clips:
            classes.append((class_dir.name, clips))

    if not classes:
        raise FileNotFoundError(f"No labeled video clips found under {VIDEO_ROOT}")

    rng = random.Random(seed)
    rng.shuffle(classes)
    if sample_count > len(classes):
        raise ValueError(
            f"Requested {sample_count} samples, but only {len(classes)} labeled "
            "sentence classes have video clips."
        )
    return [(label, rng.choice(clips)) for label, clips in classes[:sample_count]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of distinct sentence classes to sample (default: 10).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Stable sample-selection seed.")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Call Gemini for each selected public dataset clip; otherwise only list samples.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for a JSON report. Defaults to stdout only.",
    )
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be at least 1")

    try:
        samples = choose_samples(args.samples, args.seed)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    if not args.run:
        print("Dry run; no clips were sent to Gemini. Re-run with --run to evaluate.")
        for label, clip in samples:
            print(f"{label}\t{clip.relative_to(ROOT)}")
        return 0

    load_dotenv(ROOT / "backend" / ".env")
    sys.path.insert(0, str(ROOT / "backend"))
    from app.services.video_translation_service import VideoTranslationService

    translator = VideoTranslationService()
    results = []
    for index, (expected, clip) in enumerate(samples, start=1):
        print(f"[{index}/{len(samples)}] Evaluating {expected}", flush=True)
        response = translator.translate_video_file(str(clip))
        predicted = str(
            response.get("translated_sentence") or response.get("sentence") or ""
        ).strip()
        results.append(
            {
                "label": expected,
                "prediction": predicted,
                "exact_match": (
                    bool(predicted)
                    and normalized_sentence(predicted) == normalized_sentence(expected)
                ),
                "error": response.get("error"),
                "clip": str(clip.relative_to(ROOT)),
            }
        )

    translated = sum(bool(item["prediction"]) for item in results)
    correct = sum(item["exact_match"] for item in results)
    report = {
        "dataset": "ISL-CSLTR",
        "sample_count": len(results),
        "translated_count": translated,
        "abstained_or_failed_count": len(results) - translated,
        "exact_match_count": correct,
        "exact_match_accuracy": correct / len(results),
        "seed": args.seed,
        "results": results,
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Report written to {args.output}")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
