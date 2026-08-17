#!/bin/bash
# Full box-prompt sensitivity sweep for Experiment 5 (foundation side).
#
# Inference only: no training happens here, every cell re-evaluates an existing
# checkpoint under the six box perturbations. The grid is
#   zero-shot      5 models x 4 datasets                     = 20 runs
#   LoRA f=5%      4 models x 4 datasets                     = 16 runs
#   LoRA f=100%    4 models x 4 datasets                     = 16 runs
# SAM (ViT-H) is excluded from the LoRA regimes: it has no adapter checkpoint.
#
# Prerequisites: the LoRA rows need the Experiment 3 (f=0.05) and Experiment 1
# (f=1.0) foundation checkpoints on disk. The box-conditioned CNN rows of the
# reported table come from Experiment 1 instead, not from this script.
#
# A config whose output CSV already exists is skipped, so the sweep is resumable
# after an interruption. Progress is appended to results/BOXSENS_REPORT.txt.
#
# Usage:
#   bash experiments/exp5_boxsens/run_boxsens_all.sh
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

PY="${PYTHON:-python}"
RUN="experiments/exp5_boxsens/run_boxsens.py"
OUT="experiments/exp5_boxsens/results"
REPORT="${OUT}/BOXSENS_REPORT.txt"
mkdir -p "$OUT"

ZS_MODELS=(sam sam2 sam3 medsam medsam2)
LORA_MODELS=(sam2 sam3 medsam medsam2)        # SAM has no LoRA adapter
LORA_FRACTIONS=(0.05 1.0)                     # the two reported fractions
DATASETS=(ddti tn3k thyroidxl stanford_aimi)  # small -> large

echo "==================================================" | tee -a "$REPORT"
echo "BOXSENS SWEEP START $(date)" | tee -a "$REPORT"

run_one() {
    local tag="$1"; shift
    local csv="${OUT}/${tag}_boxsens.csv"
    if [ -f "$csv" ]; then
        echo "SKIP $tag (exists) $(date)" | tee -a "$REPORT"; return
    fi
    echo "START $tag $(date)" | tee -a "$REPORT"
    if "$PY" "$RUN" "$@" > "${OUT}/${tag}.log" 2>&1; then
        echo "DONE  $tag $(date)" | tee -a "$REPORT"
    else
        echo "FAIL  $tag rc=$? $(date) (see ${tag}.log)" | tee -a "$REPORT"
    fi
}

for ds in "${DATASETS[@]}"; do
    for m in "${ZS_MODELS[@]}"; do
        run_one "zeroshot_${m}_${ds}" --regime zeroshot --model "$m" --dataset "$ds"
    done
    for f in "${LORA_FRACTIONS[@]}"; do
        for m in "${LORA_MODELS[@]}"; do
            run_one "lora_${m}_${ds}_f${f}" \
                --regime lora --model "$m" --dataset "$ds" --fraction "$f"
        done
    done
done

echo "BOXSENS SWEEP COMPLETE $(date)" | tee -a "$REPORT"
echo "Next: python experiments/exp5_boxsens/aggregate_boxsens.py" | tee -a "$REPORT"
