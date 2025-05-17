# Context-Enhanced Memory-Refined Transformer for Online Action Detection

This is an implementation for our CVPR 2025 paper "[`Context-Enhanced Memory-Refined Transformer for Online Action Detection`](https://github.com/pangzhan27/cmert.io)".


## Data Preparation

You can directly download the pre-extracted feature of THUMOS and EK100 from [`TeSTra`](https://github.com/zhaoyue-zephyrus/TeSTra#pre-extracted-feature).

You can find the pre-extracted feature of CrossTask [`here`](https://github.com/DmZhukov/CrossTask)


## Training

You can train our method on THUMOS'14 by simply running:
```train
python main.py --config_file configs/THUMOS/cmert_long256_work4_kinetics_1x.yaml
```

## Evaluation

You can verify the performance of our method on THUMOS'14 by simply running:
```eval
python main.py --test 1 --config_file configs/THUMOS/cmert_long256_work4_kinetics_1x.yaml MODEL.CHECKPOINT checkpoints/THUMOS/cmert_long256_work4_kinetics_1x/epoch-9.pth MODEL.LSTR.INFERENCE_MODE batch
```
This should return the results reported in the submitted paper.




## Acknowledgements

This codebase is built upon [`TeSTra`](https://github.com/zhaoyue-zephyrus/TeSTra) and  [`MAT`](https://github.com/Echo0125/MAT-Memory-and-Anticipation-Transformer).

