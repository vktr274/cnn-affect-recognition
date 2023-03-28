# Project 2 - Convolutional Neural Network for Affect Recognition

Course: Neural Networks @ FIIT STU\
Authors: Viktor Modroczký & Michaela Hanková

## Dataset

We used the [Facial Expressions Training Data](https://www.kaggle.com/datasets/noamsegal/affectnet-training-data) from Kaggle. The dataset contains 29,042 images of 8 different emotions. The images are 96x96 pixels and are grayscale.

## Preprocessing

We created a Python script that splits the dataset into training and test sets and optionally balances the dataset. The script also creates a CSV file with label to filename mappings. The script is located in the `src` folder and is named `datasplit.py`. The script is universal and can be used for any dataset that has the same structure (dataset with a train subdirectory with images in subdirectories named after the labels).

Before running the script, the dataset should be downloaded and extracted to a trein subdirectory of a arbitrary otherwise empty folder. The script will create a `data/test` subdirectory and will move the images to the appropriate subdirectories. The script will also create a `data/train.csv` and `data/test.csv` files with label to filename mappings.

Besides the dataset, the script requires installation of required Python packages listed in the `requirements.txt` file. The packages can be installed using the following command:

`pip install -r requirements.txt`

### Running the Script

Usage:

`datasplit.py [-h] [--train-split TRAIN_SPLIT] [--seed SEED] [--label-col LABEL_COL] [--filename-col FILENAME_COL] [--global-multiplier GLOBAL_MULTIPLIER] [--balance]`

Positional argument:

- `path` - Path to the dataset that includes a train directory with the images in subdirectories named after the labels, e.g. if `path` is `data`, then the images should be in `data/train/class1`, `data/train/class2`, etc.

Options:

- `-h` - show help message and exit
- `--train-split TRAIN_SPLIT` - Train split ratio (default: 0.8)
- `--seed SEED` - Random seed (default: None)
- `--label-col LABEL_COL` - Label column name (default: label)
- `--filename-col FILENAME_COL` - Filename column name (default: filename)
- `--global-multiplier GLOBAL_MULTIPLIER` - Global multiplier for the number of images in each class (default: 1.0)
- `--balance` - Balance classes in training set (default: False)

## Model

TODO

## Training

TODO

## Results

TODO

## References

TODO
