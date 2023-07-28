# Continuous Models ∞

The following class of models treat the bootleg scores as a continuous representation. You can think of this as an image classification problem where musical noteheads are treated as individual pixels. Follow the `cnn.ipynb` notebook to train a basic 2 layer CNN. Follow instructions below to pretrain a ViT-MAE and finetune a ViT on our custom datasets.

## Pre-train ViT-MAE

Run the following command to pre-train a ViT-MAE.

```console
python3 pretrain.py  --data_path /mnt/data0/BSCRC/data/imslp_fragments.pkl \
    --output_dir ./pretrain75 --blr 1.5e-4 --weight_decay 0.05 \
    --mask_ratio 0.75 --epochs 8 --warmup_epochs 1 --batch_size 128
```

## Linear probe starting from pre-trained Vision Transformer

Run the following command to train a linear probe on top of the pre-trained ViT-MAE. Ensure to desginate the path of the pre-trained model in the `finetune` flag.

```console
python3 vit_finetune.py --probe --data_path /mnt/data0/BSCRC/data/9_way_dataset.pkl \
    --output_dir ./probe75_class9 --finetune ./pretrain75/checkpoint-7.pth --batch_size 128 \
    --nb_classes 9 --epochs 12 --warmup_epochs 1 --blr 0.1 --weight_decay 0.0 --min_lr 0
```

## Further finetuning starting from linear probe

Run the following command to further finetune the linear probe. Ensure to designate the path of the linear porbe model in the `resume` flag.

```console
python3 vit_finetune.py --probe --data_path /mnt/data0/BSCRC/data/9_way_dataset.pkl \
    --output_dir ./finetune75_class9 --finetune ./pretrain75/checkpoint-7.pth --batch_size 128 \
    --nb_classes 9 --epochs 8 --warmup_epochs 1 --blr 1e-2 --weight_decay 0.0 --min_lr 0 \
    --resume ./probe75_class9/checkpoint-11.pth
```
