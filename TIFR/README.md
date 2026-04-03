# Text-FS3DSeg

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
