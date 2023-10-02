from tqdm import tqdm
import glob
import argparse 
import re
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import Dataset
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, \
    TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.metrics import confusion_matrix

BATCH_SIZE = 64

def data_preparation(labeled_data):
    dataset = pd.read_pickle(labeled_data)
    train_X, train_y, val_X, val_y, test_X, test_y = dataset[0], dataset[1], dataset[2], dataset[3], dataset[4], dataset[5]
    
    train_df = pd.DataFrame({"text": train_X, "label": train_y})
    val_df = pd.DataFrame({"text": val_X, "label": val_y})
    test_df = pd.DataFrame({"text": test_X, "label": test_y})
    
    return train_df, val_df, test_df

def tokenizer_function(examples, tokenizer):
    return tokenizer(examples["text"], padding='max_length', truncation=True)

def label2id_function(examples, label2id):
    return {"label": [label2id[label] for label in examples["label"]]}

def test_and_eval_last_epoch(classifier_path, data_path):
    pretrained_models = glob.glob(f"{classifier_path}/*/")
    pretrained_models = [model[:-1] for model in pretrained_models]
    pretrained_models.sort(key = lambda x: int(re.search('[0-9]+$', x).group(0)))

    # Prepare data
    train_df, val_df, test_df= data_preparation(data_path)
    val_ds = Dataset.from_dict(val_df)
    test_ds = Dataset.from_dict(test_df)

    config = AutoConfig.from_pretrained(pretrained_models[0])

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_models[0])
    tokenizer.pad_token = '<pad>'
    tokenizer.model_max_length = config.n_positions
    config.pad_token_id = tokenizer.pad_token_id

    # Define label map
    label2id = {label: i for i, label in enumerate(set(train_df['label']))}

    # Tokenize and convert labels to ids
    val_ds = val_ds.map(tokenizer_function, batched=True, fn_kwargs={"tokenizer": tokenizer})
    val_ds = val_ds.map(label2id_function, batched=True, fn_kwargs={"label2id": label2id})
    test_ds = test_ds.map(tokenizer_function, batched=True, fn_kwargs={"tokenizer": tokenizer})
    test_ds = test_ds.map(label2id_function, batched=True, fn_kwargs={"label2id": label2id})

    ds_list = [val_ds, test_ds]
    ds_list = [ds.remove_columns(["text"]) for ds in ds_list]
    ds_list = [ds.rename_column("label", "labels") for ds in ds_list]
    [ds.set_format("torch") for ds in ds_list]

    val_dl = torch.utils.data.DataLoader(ds_list[0], batch_size=BATCH_SIZE)
    test_dl = torch.utils.data.DataLoader(ds_list[1], batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    progress_bar = tqdm(range(len(val_dl) + len(test_dl)))

    val_acc  = []
    test_acc = []
    val_preds = []
    test_preds = []

    pretrained_model = pretrained_models[-1] # use last epoch
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model, config=config)
    model.to(device)
    model.pad_token_id = tokenizer.pad_token_id

    correct1, correct5, total = 0, 0, 0
    model.eval()
    for batch in val_dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        val_preds.append(logits.cpu().numpy())

        _, top1 = torch.topk(logits, k=1, dim=1)
        _, top5 = torch.topk(logits, k=5, dim=1)

        total += batch['labels'].size(0)
        correct1 += torch.sum(top1 == batch['labels'][:,None]).item()
        correct5 += torch.sum(top5 == batch['labels'][:,None]).item()
        progress_bar.update(1)

    val_acc1 = correct1 / total
    val_acc5 = correct5 / total
    val_acc.append(f"{val_acc1},{val_acc5}\n")

    val_preds = np.concatenate(val_preds, axis=0)
    with open(f"{classifier_path}/val_preds.npy", "wb") as f:
        np.save(f, val_preds)
    
    with open(f"{classifier_path}/val_acc.csv", "w") as f:
        for accuracy in val_acc:
            f.write(accuracy)


    correct1, correct5, total = 0, 0, 0
    model.eval()
    for batch in test_dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        test_preds.append(logits.cpu().numpy())

        _, top1 = torch.topk(logits, k=1, dim=1)
        _, top5 = torch.topk(logits, k=5, dim=1)

        total += batch['labels'].size(0)
        correct1 += torch.sum(top1 == batch['labels'][:,None]).item()
        correct5 += torch.sum(top5 == batch['labels'][:,None]).item()
        progress_bar.update(1)

    val_acc1 = correct1 / total
    val_acc5 = correct5 / total
    test_acc.append(f"{val_acc1},{val_acc5}\n")

    test_preds = np.concatenate(test_preds, axis=0)
    with open(f"{classifier_path}/test_preds.npy", "wb") as f:
        np.save(f, test_preds)
    
    with open(f"{classifier_path}/test_acc.csv", "w") as f:
        for accuracy in test_acc:
            f.write(accuracy)


def create_confusion_matrix(predictions, labels):
    y_pred = np.argmax(predictions, axis=1)
    # label to idx 
    label2id = {label: i for i, label in enumerate(np.unique(labels))}
    y_true = [label2id[label] for label in labels]

    x_ticks = [label for label in label2id.keys()]
    y_ticks = [label for label in label2id.keys()]

    mat = confusion_matrix(y_true, y_pred)
    sns.heatmap(mat, annot=True, fmt='d', cmap='Blues', xticklabels=x_ticks, yticklabels=y_ticks)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    return mat
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate and test models (with Dense and Sparse Discrete encodings)')
    parser.add_argument('-d', '--device', help="Index of GPU to use (not parallelized)", default=0, type=int)
    parser.add_argument('-m', '--models_dir', help="Directory of models", default=None, type=str)
    parser.add_argument('-e', '--encoded_data', help="Directory of encoded data", default=None, type=str)
    args = parser.parse_args()

    torch.cuda.set_device(args.device) 
    data = Path(args.encoded_data).glob("*.pkl")
    model_base = Path(args.models_dir)

    for d in data:
        print(d.stem)
        if d.stem == "morse":
            continue

        model_path = model_base / d.stem
        print("Validation! :)")
        test_and_eval_last_epoch(model_path, d)
        print("Testing! :)")

