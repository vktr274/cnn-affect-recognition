# Project 2 - Convolutional Neural Network for Affect Recognition

Course: Neural Networks @ FIIT STU\
Authors: Viktor Modroczký & Michaela Hanková

## Dataset

We used the [Facial Expressions Training Data](https://www.kaggle.com/datasets/noamsegal/affectnet-training-data) from Kaggle. The dataset contains 29,042 images of 8 different classes of emotions which are not balanced - anger, contempt, disgust, fear, happy, neutral, sad, and surprise. The images are 96x96 pixels in size and have 3 channels (RGB).

## Preprocessing

We created a Python script that splits the dataset into training and test sets and optionally balances the dataset by augmenting classes smaller in size relative to the largest class. If balancing is enabled, it can also optionally perform global augmentation. Meaning that the number of images in each class can be increased by a global multiplier. The script also creates a CSV file with label to filename mappings for the training and test sets.

The augmentation pipleine used it the script is created using the [Albumentations](https://albumentations.ai/) library. The pipeline is a composition of transformations that are applied to the images and is defined like this:

```py
A.Compose(
  [
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(
      always_apply=True, contrast_limit=0.2, brightness_limit=0.2
    ),
    A.OneOf(
      [
        A.MotionBlur(always_apply=True),
        A.GaussNoise(always_apply=True),
        A.GaussianBlur(always_apply=True),
      ],
      p=0.5,
    ),
    A.Rotate(always_apply=True, limit=20, border_mode=cv2.BORDER_REPLICATE),
  ]
)
```

Horizontal flip is applied with a probability of 50%. Random brightness and contrast are always applied with a contrast limit of &pm;20% and a brightness limit of &pm;20%. One of motion blur, Gaussian noise, or Gaussian blur is applied with a probability of 50%. Rotation by a random angle is always applied with a limit of &pm;20 degrees and border mode set to replicate colors at the borders of the image being rotated to avoid black borders.

If enabled, augmentations are applied both to the training and test sets.

Before running the script, make sure that the dataset is downloaded and extracted to a folder with the following structure:

```text
your_dataset_folder
└── train
    ├── anger
    ├── contempt
    ├── disgust
    ├── fear
    ├── happy
    ├── neutral
    ├── sad
    └── surprise
```

The script will create a `test` subdirectory in the directory and will move part of the images from the `train` subdirectory to the `test` subdirectory. This means that the default behavior of the script is to overwrite the input directory. The script will also create a `train.csv` and `test.csv` files in the directory. The `train.csv` file will contain the label to filename mappings for the training set and the `test.csv` file will contain the label to filename mappings for the test set.

If the script is ran with a specified output path, the script will first copy the images from the `train` subdirectory in the input directory to the `train` subdirectory in the output directory and will create the `test` subdirectory in the output directory. The `train.csv` and `test.csv` files will be created in the output directory. If the output directory does not exist, it will be created. If the output directory exists, it can only contain an empty `train` subdirectory or can be empty completely.

Besides the dataset, the script requires installation of required Python packages listed in the `requirements.txt` file. The packages can be installed using the following command:

`pip install -r requirements.txt`

The script is universal and can be used for any dataset that has the same structure (dataset with a train subdirectory with images in subdirectories named after the labels).

### Running the Script

Usage:

`python datasplit.py [-h] [--balance] [--output-path OUTPUT_PATH] [--train-split TRAIN_SPLIT] [--seed SEED] [--label-col LABEL_COL] [--filename-col FILENAME_COL] [--global-multiplier GLOBAL_MULTIPLIER] [--pipeline-yaml PIPELINE_YAML] path`

Positional argument:

- `path` - Path to a directory that includes a train directory with the images in subdirectories named after the labels, e.g. if `path` is `data`, then the images should be in `data/train/class1`, `data/train/class2`, etc.

Options:

- `-h`, `--help` - show help message and exit
- `--balance` - Balance classes in training set and optionally perform global augmentation if `GLOBAL_MULTIPLIER` is greater than 1.0 (default: `False`)
- `--output OUTPUT` - Path to an empty output directory (default: `None` - overwrite input directory)
- `--train-split TRAIN_SPLIT` - Train split ratio (default: `0.8`)
- `--seed SEED` - Random seed (default: `None`)
- `--label-col LABEL_COL` - Label column name (default: `'label'`)
- `--filename-col FILENAME_COL` - Filename column name (default: `'filename'`)
- `--global-multiplier GLOBAL_MULTIPLIER` - Global multiplier for the number of images in each class (default: `1.0`). This option can be used to increase the number of images in each class but is ignored if `--balance` is not used.
- `--pipeline-yaml` - Path to a custom Albumentations Compose pipeline serialized to YAML (default: `None` - use pipeline included in this script)

If you would like to use the `--pipeline-yaml` option, the following is a brief description of how to create a custom pipeline and serialize it to YAML:

The pipeline should be serialized to YAML and has to be an instance of [`albumentations.core.composition.Compose`](https://albumentations.ai/docs/api_reference/core/composition/#albumentations.core.composition.Compose).
The pipeline can be serialized using [`albumentations.core.serialization.save`](https://albumentations.ai/docs/api_reference/core/serialization/#albumentations.core.serialization.save).

Example of serializing a custom pipeline is included in the `src` folder and is named `custom_pipeline_example.py`. Example of a serialized pipeline is included in the root folder and is named `custom_pipeline_example.yaml`.

**How we ran the script:**

`python src/datasplit.py --balance --seed 27 --output-path data_balanced data`

The resulting dataset is balanced and contains 41,008 images in total. The training set contains 32,808 images (4,101 in each class) and the test set contains 8,200 images (1,025 in each class). It's available on Kaggle [here](https://www.kaggle.com/datasets/viktormodroczky/facial-affect-dataset).

## Model

TODO

## Training

TODO

## Results

TODO

## References

TODO
