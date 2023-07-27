# Filler Classifier 🎉

Only pages in the 9-way dataset are labeled as filler and non-filler pages. In order to use all of IMSLP, we must therefore create a classifier to identify filler pages. The `notebook_01` uses our manually labeled 9-way filler pages to create a balanced dataset of filler and non-filler bootleg fragments of length 16. Then, `notebook_02` uses our pretrained LM model and finetunes a classifier to identify filler fragements. Follow directions below on how to run this classifier on all of IMSLP.

## Making Ensemble predictions on IMSLP

Before running, ensure you clone the IMSLP bootleg score repo into `data_path`

```console
git clone https://github.com/HMC-MIR/piano_bootleg_scores.git piano_bootleg_scores
```

Then, run this script from terminal via the command.

```console
python3 run_ensemble_imslp.py
```

This will run the ensemble filler classifer on all bootleg scores in IMSLP. The output is a file in `/cfg_files/ensemble_imslp.tsv.` Each row represents a single bootleg score that has the following values separated by tabs: file path to pickle file, page number, predicted filler probability (where a value of 1 represents filler page).

## Saved Predictions

Finally, `notebook_03` shows how to make ensemble predictions at the page level and saves probabilities to disk. The probabilities are then thresholded and filler pages are then reformatted as a txt file that resembles our original manually labeled 9-way filler txt file. This txt file is then used to filter out pages for our other language model training.
