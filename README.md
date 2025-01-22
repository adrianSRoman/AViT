# AViT

## Quick setup instructions

### Generate dataset file list .TXT file

```
python generate_datalist.py path/to/dataset train_dataset.txt 
```

### Change your config file for training (mainly change paths to be yours)

```
vi config/train/train.json 
```

### Run experiment

```
python train.py -C config/train/train.json
```
