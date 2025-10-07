#!/bin/bash

# Activate conda environment and run clustering script
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae
cd /home/nvidia/Documents/Hariom/InterpUseCases_autointerp/FeatureCluster
python cluster_sae_labels.py

echo "Clustering completed!"
