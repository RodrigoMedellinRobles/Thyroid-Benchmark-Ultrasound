#!/bin/bash
set -euo pipefail
# Box-conditioned nnU-Net v2 pipeline for one dataset.
#
# nnU-Net is self-configuring and drives its own plan / preprocess / train /
# predict CLI, so it does not go through run.py like U-Net and TransUNet do.
# This script runs the exact sequence used for the reported results.
#
# Box-conditioned datasets use IDs 011-014:
#   011 = DDTI, 012 = TN3K, 013 = ThyroidXL, 014 = Stanford AIMI
# Build them first with:
#   python experiments/exp1_fullsupervised/traditional_boxcond/convert_to_nnunet_boxcond.py
#
# The image goes in channel 0 (z-scored) and the tight oracle box in channel 1,
# which must stay unnormalised so it remains a clean {0,1} mask. Step 2 asserts
# the planner did not override that; if it did, the box channel would be rescaled
# and the model would silently train on a different prompt than intended.
#
# Usage:
#   bash experiments/exp1_fullsupervised/traditional_boxcond/run_nnunet_boxcond_pipeline.sh 011
#
# Training is 1000 epochs on a single fold and takes on the order of a day per
# dataset on one modern GPU. Run it under nohup/tmux, not in a notebook cell.

if [ $# -ne 1 ]; then
    echo "Usage: $0 <dataset_number (011..014)>"
    exit 1
fi

DATASET="$1"
PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON="${PYTHON:-python}"
EXP_DIR="${PROJECT_ROOT}/experiments/exp1_fullsupervised/traditional_boxcond"

NNUNET_RAW="${PROJECT_ROOT}/data/nnunet_raw"
NNUNET_PREPROCESSED="${EXP_DIR}/results/nnunet/preprocessed"
NNUNET_RESULTS="${EXP_DIR}/results/nnunet/results"
PRED_DIR="${EXP_DIR}/results/nnunet/predictions"

export nnUNet_raw="${NNUNET_RAW}"
export nnUNet_preprocessed="${NNUNET_PREPROCESSED}"
export nnUNet_results="${NNUNET_RESULTS}"

DATASET_DIR=$(ls -d "${NNUNET_RAW}/Dataset${DATASET}_"* 2>/dev/null || true)
if [ -z "${DATASET_DIR}" ]; then
    echo "Error: Dataset${DATASET}_* not found in ${NNUNET_RAW}."
    echo "Run convert_to_nnunet_boxcond.py first."
    exit 1
fi
DATASET_NAME=$(basename "${DATASET_DIR}")
echo "=============================="
echo "nnU-Net box-conditioned pipeline: ${DATASET_NAME}"
echo "=============================="

mkdir -p "${NNUNET_PREPROCESSED}" "${NNUNET_RESULTS}" "${PRED_DIR}/${DATASET_NAME}"

# 1. Plan and preprocess
echo "[1/5] plan_and_preprocess..."
nnUNetv2_plan_and_preprocess -d "${DATASET}" -c 2d --verify_dataset_integrity

# 2. Channel-normalisation check: image channel z-scored, box channel untouched.
echo "[2/5] Verifying channel normalization schemes..."
$PYTHON - "$DATASET_NAME" "${NNUNET_PREPROCESSED}" <<'PYEOF'
import json, os, sys
name, pre = sys.argv[1:3]
plan = json.load(open(os.path.join(pre, name, "nnUNetPlans.json")))
cfg = plan["configurations"]["2d"]
schemes = cfg.get("normalization_schemes") or []
print("  2d normalization_schemes:", schemes)
print("  2d use_mask_for_norm:", cfg.get("use_mask_for_norm"))
if not schemes or schemes[0] != "ZScoreNormalization":
    print(f"  ERROR: channel-0 scheme is {schemes[:1]}, expected ZScoreNormalization.")
    sys.exit(2)
if len(schemes) < 2 or schemes[1] != "NoNormalization":
    print(f"  ERROR: channel-1 (box) scheme is {schemes[1:2]}, expected NoNormalization.")
    print("  The box channel must stay a {0,1} mask; edit nnUNetPlans.json and re-run.")
    sys.exit(2)
print("  OK: channel-0 z-score + channel-1 no-normalization confirmed.")
PYEOF

# 3. Use our patient-level splits instead of nnU-Net's own random folds.
echo "[3/5] Installing patient-level splits_final.json..."
cp "${DATASET_DIR}/splits_final.json" \
   "${NNUNET_PREPROCESSED}/${DATASET_NAME}/splits_final.json"

# 4. Train fold 0, default 2D configuration
echo "[4/5] Training (fold 0, default 2d config, 1000 epochs)..."
nnUNetv2_train "${DATASET}" 2d 0

# 5. Predict on the test set (test cases carry the oracle box as channel 1)
echo "[5/5] Predicting on test set..."
nnUNetv2_predict \
    -i "${DATASET_DIR}/imagesTs" \
    -o "${PRED_DIR}/${DATASET_NAME}" \
    -d "${DATASET}" \
    -c 2d \
    -f 0

echo "Done: ${DATASET_NAME}"
echo "Predictions: ${PRED_DIR}/${DATASET_NAME}"
echo "Score them with (eval_nnunet_boxcond.py finds this directory by convention):"
echo "  python experiments/exp1_fullsupervised/traditional_boxcond/eval_nnunet_boxcond.py \\"
echo "      --dataset <ddti|tn3k|thyroidxl|stanford_aimi>"
