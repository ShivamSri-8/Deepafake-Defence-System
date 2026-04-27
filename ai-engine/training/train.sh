#!/bin/bash
# 5-PHASE TRAINING ORCHESTRATOR - Linux/Mac Shell Script
# Makes it easy to run each training phase

cd "$(dirname "$0")/../ai-engine"

if [ -z "$1" ]; then
    echo ""
    echo "========================================"
    echo "DEEPFAKE DETECTION 5-PHASE TRAINING"
    echo "========================================"
    echo ""
    echo "Usage: ./train.sh [phase]"
    echo ""
    echo "Options:"
    echo "  ./train.sh 0     = Run all phases (1-5)"
    echo "  ./train.sh 1     = PHASE 1: Data Validation"
    echo "  ./train.sh 2     = PHASE 2: Model Setup"
    echo "  ./train.sh 3     = PHASE 3: Quick Training (5-10 min)"
    echo "  ./train.sh 4     = PHASE 4: Full Training Head (2-4 hours)"
    echo "  ./train.sh 5     = PHASE 5: Fine-Tuning (4-8 hours)"
    echo ""
    echo "Examples:"
    echo "  ./train.sh 1            - Just validate data"
    echo "  ./train.sh 3            - Quick sanity check"
    echo "  ./train.sh 0            - Run everything"
    echo ""
    exit 0
fi

PHASE=$1

echo ""
echo "========================================"
echo "PHASE $PHASE TRAINING"
echo "========================================"
echo ""

case $PHASE in
    0)
        echo "Running ALL PHASES..."
        echo "Total Time: ~6-12 hours (GPU) or 3-5 days (CPU)"
        echo ""
        python training/phase_training_plan.py --phase 0
        ;;
    1)
        echo "PHASE 1: Data Validation (2-3 minutes)"
        echo "Validating 140,000 training images..."
        python training/phase_training_plan.py --phase 1
        ;;
    2)
        echo "PHASE 2: Model Architecture Setup (1-2 minutes)"
        echo "Loading Xception and building classification head..."
        python training/phase_training_plan.py --phase 2
        ;;
    3)
        echo "PHASE 3: Quick Training - Sanity Check (5-10 min GPU / 30-40 min CPU)"
        echo "Training on 1,000 small images for validation..."
        python training/phase_training_plan.py --phase 3
        ;;
    4)
        echo "PHASE 4: Full Training - Head Only (2-4 hours GPU / 1-2 days CPU)"
        echo "Training classification head on 100,000 images..."
        echo "WARNING: This will take several hours!"
        python training/phase_training_plan.py --phase 4
        ;;
    5)
        echo "PHASE 5: Fine-Tuning - Full Model (4-8 hours GPU / 2-3 days CPU)"
        echo "Fine-tuning entire model on 100,000 images..."
        echo "WARNING: This will take several hours!"
        python training/phase_training_plan.py --phase 5
        ;;
    *)
        echo "ERROR: Invalid phase number. Use 0-5."
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "PHASE $PHASE COMPLETE"
echo "========================================"
echo ""
