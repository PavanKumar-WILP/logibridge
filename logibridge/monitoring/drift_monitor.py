import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

REFERENCE_FILE = ROOT / "monitoring" / "reference_dist.json"

BINS = [0.0, 0.25, 0.50, 0.75, 1.01]

reference_scores = []

reference_distribution = None

rolling_scores = []

last_check = time.time()

reference_confidences = []
reference_distribution = None

ROLLING_WINDOW = 100
rolling_confidences = []

last_psi_time = time.time()

# ---------------------------------------------------
# Build reference distribution
# ---------------------------------------------------
def save_reference_distribution():
    global reference_confidences

    bins = [0.0, 0.25, 0.50, 0.75, 1.01]

    hist, _ = np.histogram(
        reference_confidences,
        bins=bins
    )

    distribution = (
        hist / hist.sum()
    ).tolist()

    with open(
        REFERENCE_FILE,
        "w"
    ) as f:

        json.dump(
            {
                "description": "Reference distribution of confidence scores",
                "source_samples": len(reference_confidences),
                "bins": bins,
                "distribution": distribution
            },
            f,
            indent=4
        )

    print("\n======================================")
    print("Reference distribution saved")
    print("======================================")
    print(distribution)


def add_reference_confidence(confidence):
    global reference_confidences

    reference_confidences.append(confidence)

    print(
        f"Collected {len(reference_confidences)}/300",
        end="\r"
    )

    if len(reference_confidences) >= 300:

        print()

        save_reference_distribution()

        raise SystemExit


# ---------------------------------------------------
# Load reference
# ---------------------------------------------------

def load_reference():
    global reference_distribution

    with open(REFERENCE_FILE) as f:

        reference_distribution = json.load(f)["distribution"]


# ---------------------------------------------------
# PSI
# ---------------------------------------------------

def compute_psi(expected, actual):

    expected = np.clip(expected, 1e-6, None)
    actual = np.clip(actual, 1e-6, None)

    return np.sum(
        (actual - expected)
        * np.log(actual / expected)
    )


# ---------------------------------------------------
# Monitoring
# ---------------------------------------------------

def add_confidence(score):

    global last_check

    rolling_scores.append(float(score))

    if len(rolling_scores) > ROLLING_WINDOW:

        rolling_scores.pop(0)

    if len(rolling_scores) < ROLLING_WINDOW:

        return

    if time.time() - last_check < 60:

        return

    last_check = time.time()

    hist, _ = np.histogram(
        rolling_scores,
        bins=BINS
    )

    hist = hist.astype(float)

    hist /= hist.sum()

    psi = compute_psi(
        reference_distribution,
        hist
    )

    print(
        f"\nCurrent PSI = {psi:.3f}"
    )

    if psi > 0.25:

        print(
            f"[LOGIBRIDGE DRIFT ALERT] "
            f"PSI={psi:.3f}"
        )

    elif psi < 0.10:

        print(
            "Distribution recovered "
            "(PSI < 0.10)"
        )