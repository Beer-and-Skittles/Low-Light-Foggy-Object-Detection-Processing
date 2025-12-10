#!/bin/bash

# Run training jobs in parallel
python -m scripts.train_aod_taskaware --cfg configs/aod_train_exdark.yaml
python -m scripts.train_aod_taskaware --cfg configs/aod_train_rtts.yaml
python -m scripts.train_resnet_taskaware --cfg configs/resnet_train_exdark.yaml
python -m scripts.train_resnet_taskaware --cfg configs/resnet_train_rtts.yaml
wait

echo "All jobs finished!"