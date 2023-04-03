# ResNet-50 based on the 'Deep Residual Learning for Image Recognition' paper
# by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun
# https://arxiv.org/pdf/1512.03385.pdf

import json
from typing import Tuple, Union
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras.layers import (
    Dropout,
    Rescaling,
    Conv2D,
    MaxPooling2D,
    Dense,
    GlobalAveragePooling2D,
    Add,
    BatchNormalization,
    ReLU,
    Softmax,
)


def ResidualBlockLarge(
    x_in,
    filters: Tuple[int, int, int],
    s=1,
    reduce=False,
    kernel_regularizer: Union[Regularizer, None] = None,
):
    filters1, filters2, filters3 = filters

    y_out = Conv2D(
        filters1,
        kernel_size=(1, 1),
        strides=(s, s),
        kernel_regularizer=kernel_regularizer,
    )(x_in)
    y_out = BatchNormalization()(y_out)
    y_out = ReLU()(y_out)

    y_out = Conv2D(
        filters2,
        kernel_size=(3, 3),
        padding="same",
        kernel_regularizer=kernel_regularizer,
    )(y_out)
    y_out = BatchNormalization()(y_out)
    y_out = ReLU()(y_out)

    y_out = Conv2D(filters3, kernel_size=(1, 1), kernel_regularizer=kernel_regularizer)(
        y_out
    )
    y_out = BatchNormalization()(y_out)

    if reduce:
        x_in = Conv2D(
            filters3,
            kernel_size=(1, 1),
            strides=(s, s),
            kernel_regularizer=kernel_regularizer,
        )(x_in)
        x_in = BatchNormalization()(x_in)

    y_out = Add()([y_out, x_in])

    return ReLU()(y_out)


def ResidualBlockSmall(
    x_in,
    filters: Tuple[int, int],
    s=1,
    reduce=False,
    kernel_regularizer: Union[Regularizer, None] = None,
):
    filters1, filters2 = filters

    y_out = Conv2D(
        filters1,
        kernel_size=(3, 3),
        strides=(s, s),
        padding="same",
        kernel_regularizer=kernel_regularizer,
    )(x_in)
    y_out = BatchNormalization()(y_out)
    y_out = ReLU()(y_out)

    y_out = Conv2D(
        filters2,
        kernel_size=(3, 3),
        padding="same",
        kernel_regularizer=kernel_regularizer,
    )(y_out)
    y_out = BatchNormalization()(y_out)
    y_out = ReLU()(y_out)

    if reduce:
        x_in = Conv2D(
            filters2,
            kernel_size=(1, 1),
            strides=(s, s),
            kernel_regularizer=kernel_regularizer,
        )(x_in)
        x_in = BatchNormalization()(x_in)

    y_out = Add()([y_out, x_in])

    return ReLU()(y_out)


def ResNet(
    output_units: int,
    input_shape: Tuple[int, int, int],
    block_sizes: Tuple[int, int, int, int],
    net_size: str,
    normalize=False,
    kernel_regularizer: Union[Regularizer, None] = None,
    dropout_rate=0.0,
) -> Model:
    x_in = Input(shape=input_shape)
    y_out = x_in

    if normalize:
        y_out = Rescaling(scale=1.0 / 255)(y_out)
    y_out = Conv2D(
        64, kernel_size=(7, 7), strides=(2, 2), kernel_regularizer=kernel_regularizer
    )(y_out)
    y_out = BatchNormalization()(y_out)
    y_out = ReLU()(y_out)

    y_out = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(y_out)

    block1, block2, block3, block4 = block_sizes

    for layer in range(block1):
        y_out = (
            ResidualBlockLarge(
                y_out,
                (64, 64, 256),
                s=2 if layer == 0 else 1,
                reduce=layer == 0,
                kernel_regularizer=kernel_regularizer,
            )
            if net_size == "large"
            else ResidualBlockSmall(
                y_out,
                (64, 64),
                s=1,
                reduce=False,
                kernel_regularizer=kernel_regularizer,
            )
        )

    for layer in range(block2):
        y_out = (
            ResidualBlockLarge(
                y_out,
                (128, 128, 512),
                s=2 if layer == 0 else 1,
                reduce=layer == 0,
                kernel_regularizer=kernel_regularizer,
            )
            if net_size == "large"
            else ResidualBlockSmall(
                y_out,
                (128, 128),
                s=2 if layer == 0 else 1,
                reduce=layer == 0,
                kernel_regularizer=kernel_regularizer,
            )
        )

    for layer in range(block3):
        y_out = (
            ResidualBlockLarge(
                y_out,
                (256, 256, 1024),
                s=2 if layer == 0 else 1,
                reduce=layer == 0,
                kernel_regularizer=kernel_regularizer,
            )
            if net_size == "large"
            else ResidualBlockSmall(
                y_out,
                (256, 256),
                s=2 if layer == 0 else 1,
                reduce=layer == 0,
                kernel_regularizer=kernel_regularizer,
            )
        )

    for layer in range(block4):
        y_out = (
            ResidualBlockLarge(
                y_out,
                (512, 512, 2048),
                s=2 if layer == 0 else 1,
                reduce=layer == 0,
                kernel_regularizer=kernel_regularizer,
            )
            if net_size == "large"
            else ResidualBlockSmall(
                y_out,
                (512, 512),
                s=2 if layer == 0 else 1,
                reduce=layer == 0,
                kernel_regularizer=kernel_regularizer,
            )
        )

    y_out = GlobalAveragePooling2D()(y_out)
    y_out = Dense(output_units, kernel_regularizer=kernel_regularizer)(y_out)
    y_out = Dropout(dropout_rate)(y_out)
    y_out = Softmax()(y_out)

    return Model(inputs=x_in, outputs=y_out)


def ResNet18(
    output_units: int,
    input_shape: Tuple[int, int, int],
    normalize=False,
    kernel_regularizer: Union[Regularizer, None] = None,
    dropout_rate=0.0,
) -> Model:
    return ResNet(
        output_units,
        input_shape,
        (2, 2, 2, 2),
        "small",
        normalize=normalize,
        kernel_regularizer=kernel_regularizer,
        dropout_rate=dropout_rate,
    )


def ResNet34(
    output_units: int,
    input_shape: Tuple[int, int, int],
    normalize=False,
    kernel_regularizer: Union[Regularizer, None] = None,
    dropout_rate=0.0,
) -> Model:
    return ResNet(
        output_units,
        input_shape,
        (3, 4, 6, 3),
        "small",
        normalize=normalize,
        kernel_regularizer=kernel_regularizer,
        dropout_rate=dropout_rate,
    )


def ResNet50(
    output_units: int,
    input_shape: Tuple[int, int, int],
    normalize=False,
    kernel_regularizer: Union[Regularizer, None] = None,
    dropout_rate=0.0,
) -> Model:
    return ResNet(
        output_units,
        input_shape,
        (3, 4, 6, 3),
        "large",
        normalize=normalize,
        kernel_regularizer=kernel_regularizer,
        dropout_rate=dropout_rate,
    )


def ResNet101(
    output_units: int,
    input_shape: Tuple[int, int, int],
    normalize=False,
    kernel_regularizer: Union[Regularizer, None] = None,
    dropout_rate=0.0,
) -> Model:
    return ResNet(
        output_units,
        input_shape,
        (3, 4, 23, 3),
        "large",
        normalize=normalize,
        kernel_regularizer=kernel_regularizer,
        dropout_rate=dropout_rate,
    )


def ResNet152(
    output_units: int,
    input_shape: Tuple[int, int, int],
    normalize=False,
    kernel_regularizer: Union[Regularizer, None] = None,
    dropout_rate=0.0,
) -> Model:
    return ResNet(
        output_units,
        input_shape,
        (3, 8, 36, 3),
        "large",
        normalize=normalize,
        kernel_regularizer=kernel_regularizer,
        dropout_rate=dropout_rate,
    )


def write_json_summary(model: Model, file_path: str) -> None:
    """Write a summary of the model to a JSON file.

    Args:
        model: The model to summarize.
        file_path: The path to the JSON file to write.
    """
    summary = model.to_json()
    with open(file_path, "w") as f:
        json.dump(summary, f)


if __name__ == "__main__":
    # Summarize models to test implementation.
    outputs = 8
    input_shape = (96, 96, 3)
    normalize = True

    model = ResNet18(outputs, input_shape, normalize=normalize)
    write_json_summary(model, "resnet18.json")

    model = ResNet34(outputs, input_shape, normalize=normalize)
    write_json_summary(model, "resnet34.json")

    model = ResNet50(outputs, input_shape, normalize=normalize)
    write_json_summary(model, "resnet50.json")

    model = ResNet101(outputs, input_shape, normalize=normalize)
    write_json_summary(model, "resnet101.json")

    model = ResNet152(outputs, input_shape, normalize=normalize)
    write_json_summary(model, "resnet152.json")
