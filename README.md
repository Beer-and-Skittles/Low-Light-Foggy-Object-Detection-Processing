# Split Train/Test Set
```
python -m scripts.split_data --cfg configs/default.yaml
```

# Run Baseline
## clahe
```
 python -m scripts.run_baseline_cli \
  --test_list data/splits/sample.txt \
  --method clahe \
  --class_names_json data/class_names.json \
  --device cpu
```
```
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 40/40 3.4it/s 11.6s
                   all         40        257      0.304     0.0667     0.0075    0.00272
                person          3          3     0.0355      0.667     0.0257     0.0045
               bicycle          4         11          1          0          0          0
                   car          1          1          0          0          0          0
            motorcycle         13         48          0          0          0          0
              airplane         19        138          1          0     0.0237     0.0158
                   bus          2          2          0          0          0          0
                 train          2          2          0          0    0.00547    0.00109
                 truck          3          3          0          0          0          0
         traffic light          5         11          0          0    0.00223   0.000445
          fire hydrant         10         38          1          0     0.0178    0.00535
```

## gamma
```
 python -m scripts.run_baseline_cli \
  --test_list data/splits/sample.txt \
  --method gamma \
  --class_names_json data/class_names.json \
  --device cpu
```
```
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 40/40 3.1it/s 13.0s
                   all         40        257     0.0106     0.0765    0.00626    0.00293
                person          3          3   0.000974      0.667    0.00183   0.000369
               bicycle          4         11          0          0          0          0
                   car          1          1          0          0          0          0
            motorcycle         13         48          0          0          0          0
              airplane         19        138     0.0735     0.0725     0.0447     0.0257
                   bus          2          2          0          0          0          0
                 train          2          2          0          0          0          0
                 truck          3          3          0          0          0          0
         traffic light          5         11          0          0          0          0
          fire hydrant         10         38     0.0312     0.0263      0.016    0.00321
```

## retinex
```
 python -m scripts.run_baseline_cli \
  --test_list data/splits/sample.txt \
  --method retinex \
  --class_names_json data/class_names.json \
  --device cpu
```
```
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 40/40 4.0it/s 10.1s
                   all         40        257          0          0          0          0
                person          3          3          0          0          0          0
               bicycle          4         11          0          0          0          0
                   car          1          1          0          0          0          0
            motorcycle         13         48          0          0          0          0
              airplane         19        138          0          0          0          0
                   bus          2          2          0          0          0          0
                 train          2          2          0          0          0          0
                 truck          3          3          0          0          0          0
         traffic light          5         11          0          0          0          0
          fire hydrant         10         38          0          0          0          0
```

<!-- ## hist_match
```
 python -m scripts.run_baseline_cli \
  --test_list data/splits/sample.txt \
  --method hist_match \
  --class_names_json data/class_names.json \
  --device cpu
``` -->

## Train AOD
```
 python -m train.train_aod_taskaware --cfg configs/aod_train.yaml
```

## Evaluate AOD
```
 python -m scripts.eval_aod_cli \
  --split_list data/splits/sample.txt \
  --ckpt outputs/aod_checkpoints/aod_epoch5.pt \
  --class_names_json data/class_names.json \
  --device cpu
```
```
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 40/40 2.9it/s 14.0s
                   all         40        257       0.82     0.0333     0.0189    0.00338
                person          3          3      0.202      0.333     0.0842     0.0088
               bicycle          4         11          1          0     0.0691     0.0069
                   car          1          1          0          0          0          0
            motorcycle         13         48          1          0     0.0213    0.00638
              airplane         19        138          1          0      0.015     0.0117
                   bus          2          2          1          0          0          0
                 train          2          2          1          0          0          0
                 truck          3          3          1          0          0          0
         traffic light          5         11          1          0          0          0
          fire hydrant         10         38          1          0          0          0
```

## Train ResNet
```
 python -m train.train_resnet_taskaware --cfg configs/resnet_train.yaml
```



## Evaluate ResNet
```
python -m scripts.eval_resnet_cli \
  --split_list data/splits/sample.txt \
  --ckpt outputs/resnet_checkpoints/resnet_epoch3.pt \
  --class_names_json data/class_names.json \
  --device cpu
```
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 40/40 3.6it/s 11.1s
                   all         40        257          0          0          0          0
                person          3          3          0          0          0          0
               bicycle          4         11          0          0          0          0
                   car          1          1          0          0          0          0
            motorcycle         13         48          0          0          0          0
              airplane         19        138          0          0          0          0
                   bus          2          2          0          0          0          0
                 train          2          2          0          0          0          0
                 truck          3          3          0          0          0          0
         traffic light          5         11          0          0          0          0
          fire hydrant         10         38          0          0          0          0
Speed: 3.1ms preprocess, 230.6ms inference, 0.0ms loss, 5.8ms postprocess per image
```