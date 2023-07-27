from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
import os
import glob
from utils import *
import numpy as np

L = 16
def run_ensemble_method(page):

    # classify any page with < 32 features as filler
    if page.shape[1] < 32:
        return float(1.0)

    ints = convert_binary_to_int(page)
    line = ' '.join([str(i) for i in ints])
    chars = convert_line_to_char_sequence(line)
    chars = chars.split(' ')

    frags = []
    for i in range(0, len(chars)-L, L//2):
        frags.append(' '.join(chars[i:i+L]))

    input = tokenizer(frags, return_tensors='pt', padding=True).to(device)
    preds = model(**input)
    logits = preds.logits
    probability = torch.softmax(logits, dim=1)
    return float(torch.mean(probability, 0)[1])

if __name__ == "__main__":

    torch.cuda.empty_cache()

    data_path = Path("filler")
    seed = 42
    tokenizer_path = data_path/"tokenizer/tokenizer.json"
    classifier_output_model_path = data_path/"finetune_lm"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path.parent)
    config = AutoConfig.from_pretrained(classifier_output_model_path/"epoch_7")
    config.num_labels = 2
    tokenizer.pad_token = '<pad>'
    config.pad_token_id = tokenizer.pad_token_id
    tokenizer.model_max_length = config.n_positions

    model = AutoModelForSequenceClassification.from_pretrained(classifier_output_model_path/"epoch_7")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.pad_token_id = tokenizer.pad_token_id
    model.eval()

    paths = glob.glob(str(data_path/'piano_bootleg_scores/imslp_bootleg_dir-v1/**/*.pkl'), recursive=True)

    with open("../cfg_files/ensemble_imslp.tsv", "w") as f:

        n, total = 0, len(paths)
        print(f"There are {total} bootlegs scores")

        for path in paths: # "filler/piano_bootleg_scores/imslp_bootleg_dir-v1/Charbonnet,_Alice_Ellen/Danse_des_sorci%C3%A8res_/385306.pkl"

            piece = load_pkl(path)
            root, piece_id = path.rsplit("/", 1) # "filler/piano_bootleg_scores/imslp_bootleg_dir-v1/Charbonnet,_Alice_Ellen/Danse_des_sorci%C3%A8res_", "385306.pkl"
            piece_id = piece_id.split(".")[0] # 385306
            root = root.replace("filler/piano_bootleg_scores/imslp_bootleg_dir-v1/", "") # Charbonnet,_Alice_Ellen/Danse_des_sorci%C3%A8res_

            for i, page in enumerate(piece):
                bscore = ints_to_binary_matrix(page).T == 1
                if bscore.size == 0:
                    continue
                probability = run_ensemble_method(bscore)
                f.write(f"{root}/{piece_id}\t{i}\t{probability}\n")

            n += 1
            if n % 250 == 0:
                print(f"Completed {n} pieces... {round(100 * n/total, 3)}%")
