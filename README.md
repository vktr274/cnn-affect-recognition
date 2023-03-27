# Project 2 - Convolutional Neural Network for Affect Recognition

Course: Neural Networks @ FIIT STU\
Authors: Viktor Modroczký & Michaela Hanková

## Dataset

We used the [Facial Expressions Training Data](https://www.kaggle.com/datasets/noamsegal/affectnet-training-data) from Kaggle. The dataset contains 29,042 images of 8 different emotions. The images are 96x96 pixels and are grayscale.

## Preprocessing

We created a Python script that splits the dataset into training and test sets and optionally balances the dataset. The script also creates a CSV file with label to filename mappings. The script is located in the `src` folder. The script is universal and can be used for any dataset that has the same structure (dataset with a train subdirectory with images in subdirectories named after the labels).

Usage:

`preprocess.py [-h] [--train-split TRAIN_SPLIT] [--balance] [--seed SEED] path`

Positional argument:

- `path` - Path to the dataset that includes a train directory with the images in subdirectories named after the labels, e.g. if `path` is `data`, then the images should be in `data/train/class1`, `data/train/class2`, etc.

Options:

- `-h` - show help message and exit
- `--train-split TRAIN_SPLIT` - Train split ratio (default: 0.8)
- `--balance` - Balance classes in training set (default: False)
- `--seed SEED` - Random seed (default: None)

## Model

## Training

## Results

## References
