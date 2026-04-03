# Text-FS3DSeg


## Running 

**Installation and data preparation please follow [attMPTI](https://github.com/Na-Z/attMPTI).**


### Training

Pretrain the segmentor which includes feature extractor on the available training set:

```bash
bash scripts/pretrain_segmentor.sh
```

Train our method:

```bash
bash scripts/train.sh
```

### Evaluation

Test our method:

```bash
bash scripts/eval_PAP.sh
```


## Acknowledgement
We thank [DGCNN (pytorch)](https://github.com/WangYueFt/dgcnn/tree/master/pytorch) and [attMPTI](https://github.com/Na-Z/attMPTI) for sharing their source code.
