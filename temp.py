import json
import utils
import torch
import evaluate
import transformers
import numpy as np
import pandas as pd

from torch_lr_finder import LRFinder
from pathlib import Path
from tqdm.notebook import tqdm
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.normalizers import (Sequence, Lowercase, NFD, StripAccents)
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.decoders import BPEDecoder
from transformers import AutoConfig, \
    DataCollatorWithPadding, AutoModelForSequenceClassification, \
    Trainer, TrainingArguments, AutoTokenizer, GPT2Config
from matplotlib import pyplot as plt
from sklearn.metrics import top_k_accuracy_score

import importlib
importlib.reload(utils)


def data_preparation(labeled_data):
    """Prepare data for training, validation, and testing.
    
    Returns
    -------
    train_df : pd.DataFrame
        Training data with 2 columns, "text" and "label".
    val_df : pd.DataFrame
        Validation data with 2 columns, "text" and "label".
    test_df : pd.DataFrame
        Testing data with 2 columns, "text" and "label".
    """

    # FILL IN
    # train_X, train_y, val_X, val_y, test_X, test_y = utils.load_pkl(labeled_data)
    data = utils.load_pkl(labeled_data)
    train_X, train_y, val_X, val_y, test_X, test_y = data[0], data[1], data[2], data[3], data[4], data[5]
    

    train_df = pd.DataFrame({"text": train_X, "label": train_y})
    val_df = pd.DataFrame({"text": val_X, "label": val_y})
    test_df = pd.DataFrame({"text": test_X, "label": test_y})
    
    return train_df, val_df, test_df


def tokenizer_function(examples, tokenizer):
    return tokenizer(examples["text"], padding='max_length', truncation=True)

def label2id_function(examples, label2id):
    return {"label": [label2id[label] for label in examples["label"]]}

def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")  
    predictions, labels = eval_pred
    argmaxed = np.argmax(predictions, axis=1)
    # top5_accuracy = top_k_accuracy_score(y_true=labels, y_score=predictions, k=5)
    return {"top1": accuracy.compute(predictions=argmaxed, references=labels),}
            # "top5": top5_accuracy}


def full_finetune(labeled_data, pretrained_output_model_path, classifier_output_model_path, batch_size, epochs, learning_rate):
    # Prepare data
    train_df, val_df, test_df = data_preparation(labeled_data)
    train_ds = Dataset.from_dict(train_df)
    val_ds = Dataset.from_dict(val_df)
    test_ds = Dataset.from_dict(test_df)

    # Define label map
    label2id = {label: i for i, label in enumerate(set(train_df['label']))}
    id2label = {i: label for label, i in label2id.items()}

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_output_model_path)
    tokenizer.pad_token = '<pad>'


    # Define model
    config = AutoConfig.from_pretrained(pretrained_output_model_path)
    config.num_labels = len(label2id)
    config.pad_token_id = tokenizer.pad_token_id
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_output_model_path, config=config)
    tokenizer.model_max_length = config.n_positions

    # Tokenize and convert labels to ids
    train_ds = train_ds.map(tokenizer_function, batched=True, fn_kwargs={"tokenizer": tokenizer})
    val_ds = val_ds.map(tokenizer_function, batched=True, fn_kwargs={"tokenizer": tokenizer})
    train_ds = train_ds.map(label2id_function, batched=True, fn_kwargs={"label2id": label2id})
    val_ds = val_ds.map(label2id_function, batched=True, fn_kwargs={"label2id": label2id})

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest", max_length=1024)

    # Freeze all layers except the classifier
    for name, param in model.named_parameters():
        param.requires_grad = True
    model.score.weight.requires_grad = True


    # Define training arguments
    training_args = TrainingArguments(
        output_dir=classifier_output_model_path,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train model
    trainer.train()


data = Path("/mnt/data0/BSCRC/data/9_way_encoded/").glob("*.pkl")
model_base = Path("/home/ajain/ttmp/EWLLMs/experiments/9_way_linear_probes/")
finetune_base = Path("/home/ajain/ttmp/EWLLMs/experiments/9_way_full_finetunes")
finetune_base.mkdir(exist_ok=True, parents=True)

for d in data:
    if d.stem == "morse":
        continue
    print(d.stem)
    pretrained = model_base/d.stem/"checkpoint-5256"
    output = finetune_base/d.stem
    output.mkdir(exist_ok=True, parents=True)
    full_finetune(d, pretrained, output, batch_size=16, epochs=8, learning_rate=5e-5)
    