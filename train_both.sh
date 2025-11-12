#!/bin/bash

echo "Training ResNet with Attention..."
python train.py --model_type attention

echo ""
echo "Training Baseline ResNet50..."
python train.py --model_type baseline

echo ""
echo "================================================"
echo "Both models trained!"
echo "Go to https://app.clear.ml"
echo "Project: ResNet_Attention_Comparison (or param in config.py)"
echo "================================================"
