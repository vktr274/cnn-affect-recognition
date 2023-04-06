# Project 2 - Convolutional Neural Network for Affect Recognition

Course: Neural Networks @ FIIT STU\
Authors: Viktor Modroczký & Michaela Hanková

## Dataset

We used the [Facial Expressions Training Data](https://www.kaggle.com/datasets/noamsegal/affectnet-training-data) from Kaggle. The dataset contains 29,042 images of 8 different classes of emotions which are not balanced - anger, contempt, disgust, fear, happy, neutral, sad, and surprise. The images are 96x96 pixels in size and have 3 channels (RGB).

## Preprocessing

### Datasplit Script

We created a Python script that splits the dataset into training and test sets by random sampling from the original dataset and optionally balances the dataset by augmenting classes smaller in size relative to the largest class. If balancing is enabled, it can also optionally perform global augmentation. Meaning that the number of images in each class can be increased by a global multiplier. The script also creates a CSV file with label to filename mappings for the training and test sets.

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
    A.PixelDropout(p=0.25),
    A.Rotate(always_apply=True, limit=20, border_mode=cv2.BORDER_REPLICATE),
  ]
)
```

Horizontal flip is applied with a probability of 50%. Random brightness and contrast are always applied with a contrast limit of &pm;20% and a brightness limit of &pm;20%. One of motion blur, Gaussian noise, or Gaussian blur is applied with a probability of 50%. Pixel dropout is applied with a probability of 25%. Rotation by a random angle is always applied with a limit of &pm;20 degrees and border mode set to replicate colors at the borders of the image being rotated to avoid black borders.

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

Besides the dataset, the script requires installation of required Python packages listed in the [`requirements.txt`](./requirements.txt) file. The packages can be installed using the following command:

`pip install -r requirements.txt`

The script is universal and can be used for any dataset that has the same structure (dataset with a train subdirectory with images in subdirectories named after the labels).

### Running the Datasplit Script

The script is available in the [`src/datasplit.py`](./src/datasplit.py) file. It can be run following this pattern:

`python datasplit.py [-h] [--balance-train] [--balance-test] [--output-path OUTPUT_PATH] [--train-split TRAIN_SPLIT] [--seed SEED] [--label-col LABEL_COL] [--filename-col FILENAME_COL] [--global-multiplier GLOBAL_MULTIPLIER] [--pipeline-yaml PIPELINE_YAML] path`

Positional argument:

- `path` - Path to a directory that includes a train directory with the images in subdirectories named after the labels, e.g. if `path` is `data`, then the images should be in `data/train/class1`, `data/train/class2`, etc.

Options:

- `-h`, `--help` - show help message and exit
- `--balance-train` - Balance classes in training set and optionally perform global augmentation for the training set if `GLOBAL_MULTIPLIER` is greater than 1.0 (default: `False`)
- `--balance-test` - Balance classes in created test set and optionally perform global augmentation for the test set if `GLOBAL_MULTIPLIER` is greater than 1.0 (default: `False`)
- `--output OUTPUT` - Path to an empty output directory (default: `None` - overwrite input directory)
- `--train-split TRAIN_SPLIT` - Train split ratio (default: `0.8`)
- `--seed SEED` - Random seed (default: `None`)
- `--label-col LABEL_COL` - Label column name (default: `'label'`)
- `--filename-col FILENAME_COL` - Filename column name (default: `'filename'`)
- `--global-multiplier GLOBAL_MULTIPLIER` - Global multiplier for the number of images in each class (default: `1.0`). This option can be used to increase the number of images in each class but is ignored if `--balance` is not used.
- `--pipeline-yaml` - Path to a custom Albumentations Compose pipeline serialized to YAML (default: `None` - use pipeline included in this script)

If you would like to use the `--pipeline-yaml` option, the following is a brief description of a custom pipeline and its serialization:

The pipeline has to be an instance of [`albumentations.core.composition.Compose`](https://albumentations.ai/docs/api_reference/core/composition/#albumentations.core.composition.Compose) and it must be serialized to a YAML file using [`albumentations.core.serialization.save`](https://albumentations.ai/docs/api_reference/core/serialization/#albumentations.core.serialization.save). The script will then internally be able to deserialize the pipeline using [`albumentations.core.serialization.load`](https://albumentations.ai/docs/api_reference/core/serialization/#albumentations.core.serialization.load).

Example of serializing a custom pipeline is included in the [`src`](./src) folder and is named [`custom_pipeline_example.py`](./src/custom_pipeline_example.py). Example of a serialized pipeline is included in the root folder and is named [`custom_pipeline_example.yml`](./custom_pipeline_example.yml).

### Relabeling the Dataset

The original dataset also has a `labels.csv` file which includes improved labels by machine learning. We decided to create a script that relabels the dataset using the improved labels. The script is available in the [`src/relabel.py`](./src/relabel.py) file.

### Running the Relabeling Script

It can be run following this pattern:

`python relabel.py [-h] [--output-path OUTPUT_PATH] [--label-csv] path`

Positional argument:

- `path` - Path to a directory that includes a train directory with the images in subdirectories named after the labels, e.g. if `path` is `data`, then the images should be in `data/train/class1`, `data/train/class2`, etc.

Options:

- `-h`, `--help` - show help message and exit

- `--output-path OUTPUT_PATH` - Path to an output directory (required)

- `--label-csv LABEL_CSV` - Path to a CSV file with improved labels (default: `labels.csv` - use labels from `labels.csv`)

### How we ran the scripts

We ran the datasplit script for the original dataset and relabeled dataset with and without balancing the classes.

The following commands were used to create the datasets:

`python src/datasplit.py --seed 27 --output-path data_split data`

The resulting dataset is not balanced but only split into training and test sets and contains 29,042 images in total. The training set contains 23,234 images and the test set contains 5,808 images.

The dataset is available on Kaggle: [Facial Affect Dataset Unbalanced](https://www.kaggle.com/datasets/viktormodroczky/facial-affect-dataset-unbalanced)

`python src/datasplit.py --balance-train --balance-test --seed 27 --output-path data_balanced_1x data`

The resulting dataset is balanced and contains 41,008 images in total. The training set contains 32,808 images (4,101 in each class) and the test set contains 8,200 images (1,025 in each class).

The balanced dataset is available on Kaggle: [Facial Affect Dataset](https://www.kaggle.com/datasets/viktormodroczky/facial-affect-dataset)

`python src/datasplit.py --seed 27 --output-path data_relabeled_split data_relabeled`

The resulting dataset is not balanced but only split into training and test sets and contains 28,664 images in total. The training set contains 22,930 images and the test set contains 5,734 images. The dataset is smaller due to the fact that the `relabel.py` script overwrites images with the same filename (issue in the script was discovered after the dataset was created).

The dataset is available on Kaggle: [Facial Affect Dataset Relabeled Unbalanced](https://www.kaggle.com/datasets/viktormodroczky/facial-affect-dataset-relabeled-unbalanced)

`python src/datasplit.py --balance-train --balance-test --seed 27 --output-path data_relabeled_balanced_1x data_relabeled`

The resulting dataset is not balanced but only split into training and test sets and contains 36,514 images in total. The training set contains 29,217 images and the test set contains 7,297 images. Again, the dataset is smaller due to the fact that the `relabel.py` script overwrites images with the same filename.

The dataset is available on Kaggle: [Facial Affect Dataset Relabeled](https://www.kaggle.com/datasets/viktormodroczky/facial-affect-data-relabeled)

## Model

We decided to implement each ResNet model in Tensorflow and choose the best performing model as our final model. The models are based on the [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) paper by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun (2015). The models are implemented in the [`src/resnets.py`](./src/resnets.py) file.

We used the following ResNet models:

- ResNet-18
- ResNet-34
- ResNet-50
- ResNet-101
- ResNet-152

Each model is implemented as a function that follows this signature:

```py
def ResNetN(
    output_units: int,
    input_shape: Tuple[int, int, int],
    normalize=False,
    kernel_regularizer: Union[Regularizer, None] = None,
    kernel_initializer="he_uniform",
    dropout_rate=0.0,
) -> Model
```

where N in the function name is the number of layers in the model that should be replaced with 18, 34, 50, 101, or 152. The function takes the following parameters:

- `output_units` - Number of output units (number of classes)
- `input_shape` - Shape of the input images
- `normalize` - Whether to normalize the input images to the range [0, 1] (default: `False`)
- `kernel_regularizer` - Kernel regularizer of class `tf.keras.regularizers.Regularizer` (default: `None`)
- `kernel_initializer` - Kernel initializer (default: `he_uniform`)
- `dropout_rate` - Dropout rate used after global average pooling (default: `0.0`)

Models use the [Functional API](https://www.tensorflow.org/guide/keras/functional) of Keras under the hood which defines the model's structure as a directed acyclic graph of layers. The function returns a `tf.keras.Model` instance that needs to be compiled and trained.

The implementation was tested by successfully creating each model and printing its summary to text files. We used a helper function to print model summaries to text files:

```py
def write_summary(model: Model, file_path: str) -> None:
    with open(file_path, "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))
```

The text files are included in the root folder and are named [`resnet18.txt`](./resnet18.txt), [`resnet34.txt`](./resnet34.txt), [`resnet50.txt`](./resnet50.txt), [`resnet101.txt`](./resnet101.txt), and [`resnet152.txt`](./resnet152.txt).

## Training and Testing Environment

We used Kaggle notebooks for model training and testing on a GPU - a Tesla P100. Training complex model like ResNets on a CPU would take a very long time so we didn't prepare a Docker image for CPU training and testing.

## Training

As a starting point we were inspired by Kaiming He et al. and their paper. They used SGD with a learning rate of 0.1 and a momentum of 0.9. They also used weight decay of 0.0001 and a batch size of 256. We decided to use all but the batch size, which we set to 32. We note that instead of weight decay we used a kernel regularizer with an L2 norm of 0.0001, which according to Ilya Loshchilov and Frank Hutter ([Fixing Weight Decay Regularization in Adam](https://arxiv.org/pdf/1711.05101v2.pdf), 2018) is equivalent to weight decay when paired with SGD.

We also used a similar learning rate schedule as in the paper. Our scheduler was set to monitor validation loss and reduce the learning rate by a factor of 0.1 when the validation loss did not improve for 5 epochs. We also used early stopping with a patience of 10 epochs and best weights restoration.

These hyperparameters were used for all ResNet models we implemented. As training data we used the unbalanced datasets first to get a baseline on how well the models perform. All models were trained on the dataset with original labels and the relabeled dataset so we could compare the effect of relabeling the dataset on the model performance.

All images were normalized to the range [0, 1] using a `Rescaling` layer with a scale of 1.0/255. The training data was split into training and validation sets using a 80/20 split.

Training has been logged using [Weights & Biases](https://wandb.ai/). The training notebooks are available in the [`src`](./src) folder. The notebook [`resnet-selection-original-labels.ipynb`](./src/resnet-selection-original-labels.ipynb) contains the training of the models on the unbalanced dataset with original labels. The notebook [`resnet-selection-improved-labels.ipynb`](./src/resnet-selection-improved-labels.ipynb) contains the training of the models on the unbalanced dataset with improved labels.

The report from training with the unbalanced dataset with original labels and without any augmentation is available on Weights & Biases [here](https://api.wandb.ai/links/nsiete23/5wrwkqey).

The report from training with the unbalanced dataset with improved labels and without any augmentation is available on Weights & Biases [here](https://api.wandb.ai/links/nsiete23/cmp2nhvu).

In both cases the ResNet-18 and ResNet-34 seem to perform the best even though every model struggled to learn anything. Only the training accuracy kept increasing while the validation accuracy reached a plateau over time. The smaller models reached this validation accuracy plateau earlier than the larger models. After reaching the plateau on ResNet-18 and ResNet-34 the models started to visibly overfit because of earlier plateauing. The larger models started overfitting too but it was not as visible on the Wandb validation loss graphs due to large validation loss at earlier epochs. After zooming in on the validation loss graphs we could see increasing validation loss on the larger models as well.

### ResNet-18 Training

After seeing the baseline results, we decided to continue using ResNet-18 not only because it performed relatively well compared to the other models, but also because it is the smallest model and therefore the fastest to train considering our time constraints.

The next step was to try to improve the model performance by using the balanced dataset with improved labels. The balancing was done by oversampling the minority classes as described in the [Preprocessing](#preprocessing) section.
We switched from SGD with momentum to Adam. L2 regularization was omitted because according to Ilya Loshchilov and Frank Hutter it is not effective when using the Adam optimizer. Since Adam is an optimizer with adaptive learning rate, we also omitted the learning rate scheduler. We continued using early stopping with a patience of 10 epochs and best weights restoration.

At first we chose 64 as the batch size and 0.0001 as the learning rate for the Adam optimizer. The training went very poorly so we ended the training on epoch 7 with a training loss of 0.0975 and a validation loss of 3.4805 which was higher than the starting validation loss of 1.7675.

We decided to try a much smaller learning rate of 0.000001 keeping the batch size at 64. The model didn't overfit before we stopped the training after 50 epochs but the validation loss was still high (1.7125) and kept significantly diverging from the training loss (1.0769).

Next we tried a learning rate of 0.00000001 and kept the batch size.

## Results

TODO

## References

[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun (2015)

[Fixing Weight Decay Regularization in Adam](https://arxiv.org/pdf/1711.05101v2.pdf) by Ilya Loshchilov and Frank Hutter (2018)

[Tensorflow Docs](https://www.tensorflow.org/api_docs/python)
