# Project 2 - Convolutional Neural Network for Affect Recognition

Course: Neural Networks @ FIIT STU\
Authors: Viktor Modroczký & Michaela Hanková

## Dataset

We used the [Facial Expressions Training Data](https://www.kaggle.com/datasets/noamsegal/affectnet-training-data) from Kaggle. The dataset contains 29,042 images of 8 different emotions. The images are 96x96 pixels and are grayscale.

## Preprocessing

We created a Python script that splits the dataset into training and test sets and optionally balances the dataset. The script also creates a CSV file with label to filename mappings. The script is located in the `src` folder and is named `datasplit.py`.

Before running the script, the dataset should be downloaded and extracted to a train subdirectory of an arbitrary otherwise empty folder. The script will create a `test` subdirectory in the folder and will move the images to the appropriate subdirectories. The script will also create a `train.csv` and `test.csv` files in the folder. The `train.csv` file will contain the label to filename mappings for the training set and the `test.csv` file will contain the label to filename mappings for the test set.

Besides the dataset, the script requires installation of required Python packages listed in the `requirements.txt` file. The packages can be installed using the following command:

`pip install -r requirements.txt`

The script is universal and can be used for any dataset that has the same structure (dataset with a train subdirectory with images in subdirectories named after the labels).

### Running the Script

Usage:

`datasplit.py [-h] [--train-split TRAIN_SPLIT] [--seed SEED] [--label-col LABEL_COL] [--filename-col FILENAME_COL] [--global-multiplier GLOBAL_MULTIPLIER] [--balance]`

Positional argument:

- `path` - Path to the dataset that includes a train directory with the images in subdirectories named after the labels, e.g. if `path` is `data`, then the images should be in `data/train/class1`, `data/train/class2`, etc.

Options:

- `-h` - show help message and exit
- `--train-split TRAIN_SPLIT` - Train split ratio (default: `0.8`)
- `--seed SEED` - Random seed (default: `None`)
- `--label-col LABEL_COL` - Label column name (default: `'label'`)
- `--filename-col FILENAME_COL` - Filename column name (default: `'filename'`)
- `--global-multiplier GLOBAL_MULTIPLIER` - Global multiplier for the number of images in each class (default: `1.0`). This option can be used to increase the number of images in each class but is ignored if `--balance` is not used.
- `--balance` - Balance classes in training set and optionally perform global augmentation if `GLOBAL_MULTIPLIER` is greater than 1.0 (default: `False`)

## Model

TODO

## Training

TODO

## Results

TODO

## References

TODO
