"""Scam Shield threshold tuning script.

Standalone, run manually — NOT part of the pytest suite.
This is a one-time calibration tool that makes real Gemini API calls.

Usage:
    cd hifazat
    python scripts/tune_scam_threshold.py

Requires:
    GEMINI_API_KEY in .env or environment
    tests/fixtures/scam_eval_set.json (the labeled eval set)

Rate limiting: 4-second sleep between calls stays comfortably
under Gemini's 5 RPM free-tier limit for gemini-2.5-flash. A 14-example eval set
completes in ~3 minutes.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

# Add project root to path so we can import app modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

CANDIDATE_THRESHOLDS = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
EVAL_SET_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "scam_eval_set.json"
SLEEP_BETWEEN_CALLS = 12  # seconds — stay under 5 RPM free-tier limit for gemini-2.5-flash


def score_at_threshold(
    results: list[tuple[bool, float]],
    threshold: float,
) -> tuple[float, float, float]:
    """Compute precision, recall, F1 at the given threshold.

    results: list of (ground_truth_is_scam, model_confidence)
    A message is "predicted scam" if confidence >= threshold.
    """
    tp = fp = fn = tn = 0
    for is_scam, confidence in results:
        predicted_scam = confidence >= threshold
        if is_scam and predicted_scam:
            tp += 1
        elif is_scam and not predicted_scam:
            fn += 1
        elif not is_scam and predicted_scam:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


async def run() -> None:
    """Run the eval set against real Gemini, print precision/recall table."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set. Add it to .env or environment.")
        sys.exit(1)

    if not EVAL_SET_PATH.exists():
        print(f"ERROR: Eval set not found at {EVAL_SET_PATH}")
        sys.exit(1)

    from app.llm.provider import GeminiProvider

    provider = GeminiProvider(api_key=api_key)
    eval_set = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))

    print(f"Running {len(eval_set)} examples against Gemini...")
    print(f"Sleep between calls: {SLEEP_BETWEEN_CALLS}s (rate limit protection)")
    print("-" * 70)

    results: list[tuple[bool, float]] = []
    errors = 0

    for i, example in enumerate(eval_set):
        label = "SCAM" if example["is_scam"] else "SAFE"
        pattern = example.get("pattern", "")
        try:
            analysis = await provider.analyze_scam(example["text"])
            conf = analysis.confidence
            is_scam_pred = "SCAM" if analysis.is_scam else "SAFE"
            results.append((example["is_scam"], conf))
            print(
                f"  [{i+1:2d}/{len(eval_set)}] "
                f"truth={label:4s}  pred={is_scam_pred:4s}  conf={conf:.2f}  "
                f"({pattern})"
            )
        except Exception as e:
            errors += 1
            print(f"  [{i+1:2d}/{len(eval_set)}] ERROR: {e}  ({pattern})")
            # Still append with 0.0 confidence so metrics aren't skewed by missing data
            results.append((example["is_scam"], 0.0))

        if i < len(eval_set) - 1:
            await asyncio.sleep(SLEEP_BETWEEN_CALLS)

    print("-" * 70)
    if errors:
        print(f"WARNING: {errors} API errors. Results may be noisy.")
    print()

    # Print threshold table
    print(f"{'threshold':>10}  {'precision':>10}  {'recall':>8}  {'f1':>6}")
    print("-" * 40)

    best_f1 = 0.0
    best_threshold = 0.0

    for threshold in CANDIDATE_THRESHOLDS:
        precision, recall, f1 = score_at_threshold(results, threshold)
        marker = ""
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            marker = "  <-- best"
        print(
            f"  {threshold:>8.2f}  {precision:>10.2f}  {recall:>8.2f}  {f1:>6.2f}{marker}"
        )

    print("-" * 40)
    print(f"\nRecommended threshold: {best_threshold} (F1={best_f1:.2f})")
    print(
        f"\nTo apply: update SCAM_CONFIDENCE_THRESHOLD in "
        f"app/agents/scam_shield.py to {best_threshold}"
    )

    # Save raw results for reproducibility
    results_path = Path(__file__).resolve().parent / "tuning_results.json"
    results_data = {
        "results": [
            {"is_scam": r[0], "confidence": r[1]} for r in results
        ],
        "thresholds": {
            str(t): {
                "precision": score_at_threshold(results, t)[0],
                "recall": score_at_threshold(results, t)[1],
                "f1": score_at_threshold(results, t)[2],
            }
            for t in CANDIDATE_THRESHOLDS
        },
        "recommended_threshold": best_threshold,
        "errors": errors,
    }
    results_path.write_text(json.dumps(results_data, indent=2), encoding="utf-8")
    print(f"Raw results saved to {results_path}")


if __name__ == "__main__":
    asyncio.run(run())
