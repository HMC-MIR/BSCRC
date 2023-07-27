# BSCRC
Bootleg Score Composer Recognition Challenge 🏆


## Datasets
To regenerate the datasets, navigate to `dataset_creation.ipynb` and change the **data_path** variable to wherever you would like to save the dataset files. Run the notebook all the way through to get your own `9_way_dataset.pkl`and `100_way_dataset.pkl`. *Note that regenerating the datasets requires you to have the predictions from the filler classifier, like `ensemble_imslp.tsv` found in the filler folder.

The 9-way dataset has 9 hand-selected composers for being generally well known, whereas the 100-way composers has the top 100 composers with the most registered bootleg score events. You can see the lists for both sets of composers in `config`.

When unpickled, these files store a tuple in the format of (x_train, y_train, meta_train, x_valid, y_valid, meta_valid, x_test, y_test, meta_test).

"x" represents the input features formatted as a list of numpy arrays.

"y" represents the labels formated as a list of integers. The integer values is assigned by the index of the composer after being sorting all of them alphabetically.

"meta" represents the metadata for the fragment, formatted as a tuple of (ID, start_offset). The ID represents the unique ID assigned to the PDF the fragment was grabbed from on the IMSLP archive. The start offset is the starting index of the fragment in the overall bootleg score of the PDF.
