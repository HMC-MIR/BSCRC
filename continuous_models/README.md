# Continuous Models ∞

The following class of models treat the bootleg scores as a continuous representation. You can think of this as an image classification problem where musical noteheads are treated as individual pixels. Follow the `cnn.ipynb` notebook to train a basic 2 layer CNN. Follow instructions below to pretrain a ViT-MAE and finetune a ViT on our custom datasets.

## Pre-train ViT-MAE

```console
python3 pretrain.py  --data_path /mnt/data0/BSCRC/data/imslp_fragments.pkl \
    --output_dir ./pretrain75 --blr 1.5e-4 --weight_decay 0.05 \
    --mask_ratio 0.75 --epochs 8 --warmup_epochs 1 --batch_size 128
```

## Linear probe starting from pre-trained Vision Transformer

```console
python3 finetune.py --probe --data_path /mnt/data0/Datasets/9_way_dataset.pkl \
    --output_dir ./probe75 --finetune ./pretrain75/checkpoint-7.pth --batch_size 128 \
    --nb_classes 9 --epochs 12 --warmup_epochs 1 --blr 0.1 --weight_decay 0.0 --min_lr 0
```

## Further finetuning starting from linear probe

