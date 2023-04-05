# ResNet-18, ResNet-34, ResNet-50, ResNet-101, and ResNet-152
# based on the 'Deep Residual Learning for Image Recognition' paper
# by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun
# https://arxiv.org/pdf/1512.03385.pdf

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
    """
    Create a ResNet block with 3 layers

    :param x_in: input tensor
    :param filters: number of filters in each layer
    :param s: stride used when reducing the input tensor
    :param reduce: whether to reduce the input tensor
    :param kernel_regularizer: kernel regularizer

    :return: output tensor
    """
    filters1, filters2, filters3 = filters

    y_out = Conv2D(
        filters1,
        kernel_size=(1, 1),
        strides=(s, s),
        kernel_regularizer=kernel_regularizer,
        kernel_initializer="he_uniform",
    )(x_in)
    y_out = BatchNormalization()(y_out)
    y_out = ReLU()(y_out)

    y_out = Conv2D(
        filters2,
        kernel_size=(3, 3),
        padding="same",
        kernel_regularizer=kernel_regularizer,
        kernel_initializer="he_uniform",
    )(y_out)
    y_out = BatchNormalization()(y_out)
    y_out = ReLU()(y_out)

    y_out = Conv2D(
        filters3,
        kernel_size=(1, 1),
        kernel_regularizer=kernel_regularizer,
        kernel_initializer="he_uniform",
    )(y_out)
    y_out = BatchNormalization()(y_out)

    if reduce:
        x_in = Conv2D(
            filters3,
            kernel_size=(1, 1),
            strides=(s, s),
            kernel_regularizer=kernel_regularizer,
            kernel_initializer="he_uniform",
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
    """
    Create a ResNet block with 2 layers

    :param x_in: input tensor
    :param filters: number of filters in each layer
    :param s: stride used when reducing the input tensor
    :param reduce: whether to reduce the input tensor

    :return: output tensor
    """
    filters1, filters2 = filters

    y_out = Conv2D(
        filters1,
        kernel_size=(3, 3),
        strides=(s, s),
        padding="same",
        kernel_regularizer=kernel_regularizer,
        kernel_initializer="he_uniform",
    )(x_in)
    y_out = BatchNormalization()(y_out)
    y_out = ReLU()(y_out)

    y_out = Conv2D(
        filters2,
        kernel_size=(3, 3),
        padding="same",
        kernel_regularizer=kernel_regularizer,
        kernel_initializer="he_uniform",
    )(y_out)
    y_out = BatchNormalization()(y_out)
    y_out = ReLU()(y_out)

    if reduce:
        x_in = Conv2D(
            filters2,
            kernel_size=(1, 1),
            strides=(s, s),
            kernel_regularizer=kernel_regularizer,
            kernel_initializer="he_uniform",
        )(x_in)
        x_in = BatchNormalization()(x_in)

    y_out = Add()([y_out, x_in])

    return ReLU()(y_out)


def ResNet(
    output_units: int,
    input_shape: Tuple[int, int, int],
    block_sizes: Tuple[int, int, int, int],
    net_size: str,
    include_top=True,
    normalize=False,
    kernel_regularizer: Union[Regularizer, None] = None,
    dropout_rate=0.0,
) -> Model:
    """
    Create one of ResNet-18, ResNet-34, ResNet-50, ResNet-101, and ResNet-152

    :param output_units: number of output units
    :param input_shape: input shape
    :param block_sizes: number of layers in each block
    :param net_size: 'small' or 'large'
    :param normalize: whether to normalize the input
    :param kernel_regularizer: kernel regularizer
    :param dropout_rate: dropout rate

    :return: ResNet model
    """
    x_in = Input(shape=input_shape)
    y_out = x_in

    if normalize:
        y_out = Rescaling(scale=1.0 / 255)(y_out)
    y_out = Conv2D(
        64,
        kernel_size=(7, 7),
        strides=(2, 2),
        kernel_regularizer=kernel_regularizer,
        kernel_initializer="he_uniform",
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
    if dropout_rate > 0.0:
        y_out = Dropout(dropout_rate)(y_out)

    if include_top:
        y_out = Dense(
            output_units,
            kernel_regularizer=kernel_regularizer,
            kernel_initializer="he_uniform",
        )(y_out)
        y_out = Softmax()(y_out)

    return Model(inputs=x_in, outputs=y_out)


def ResNet18(
    output_units: int,
    input_shape: Tuple[int, int, int],
    include_top=True,
    normalize=False,
    kernel_regularizer: Union[Regularizer, None] = None,
    dropout_rate=0.0,
) -> Model:
    """
    Create a ResNet-18 model.

    :param output_units: The number of output units.
    :param input_shape: The shape of the input.
    :param normalize: Whether to normalize the input.
    :param kernel_regularizer: The kernel regularizer to use.
    :param dropout_rate: The dropout rate to use.

    :return: The model.
    """
    return ResNet(
        output_units,
        input_shape,
        (2, 2, 2, 2),
        "small",
        include_top=include_top,
        normalize=normalize,
        kernel_regularizer=kernel_regularizer,
        dropout_rate=dropout_rate,
    )


def ResNet34(
    output_units: int,
    input_shape: Tuple[int, int, int],
    include_top=True,
    normalize=False,
    kernel_regularizer: Union[Regularizer, None] = None,
    dropout_rate=0.0,
) -> Model:
    """
    Create a ResNet-34 model.

    :param output_units: The number of output units.
    :param input_shape: The shape of the input.
    :param normalize: Whether to normalize the input.
    :param kernel_regularizer: The kernel regularizer to use.
    :param dropout_rate: The dropout rate to use.

    :return: The model.
    """
    return ResNet(
        output_units,
        input_shape,
        (3, 4, 6, 3),
        "small",
        include_top=include_top,
        normalize=normalize,
        kernel_regularizer=kernel_regularizer,
        dropout_rate=dropout_rate,
    )


def ResNet50(
    output_units: int,
    input_shape: Tuple[int, int, int],
    include_top=True,
    normalize=False,
    kernel_regularizer: Union[Regularizer, None] = None,
    dropout_rate=0.0,
) -> Model:
    """
    Create a ResNet-50 model.

    :param output_units: The number of output units.
    :param input_shape: The shape of the input.
    :param normalize: Whether to normalize the input.
    :param kernel_regularizer: The kernel regularizer to use.
    :param dropout_rate: The dropout rate to use.

    :return: The model.
    """
    return ResNet(
        output_units,
        input_shape,
        (3, 4, 6, 3),
        "large",
        include_top=include_top,
        normalize=normalize,
        kernel_regularizer=kernel_regularizer,
        dropout_rate=dropout_rate,
    )


def ResNet101(
    output_units: int,
    input_shape: Tuple[int, int, int],
    include_top=True,
    normalize=False,
    kernel_regularizer: Union[Regularizer, None] = None,
    dropout_rate=0.0,
) -> Model:
    """
    Create a ResNet-101 model.

    :param output_units: The number of output units.
    :param input_shape: The shape of the input.
    :param normalize: Whether to normalize the input.
    :param kernel_regularizer: The kernel regularizer to use.
    :param dropout_rate: The dropout rate to use.

    :return: The model.
    """
    return ResNet(
        output_units,
        input_shape,
        (3, 4, 23, 3),
        "large",
        include_top=include_top,
        normalize=normalize,
        kernel_regularizer=kernel_regularizer,
        dropout_rate=dropout_rate,
    )


def ResNet152(
    output_units: int,
    input_shape: Tuple[int, int, int],
    include_top=True,
    normalize=False,
    kernel_regularizer: Union[Regularizer, None] = None,
    dropout_rate=0.0,
) -> Model:
    """
    Create a ResNet-152 model.

    :param output_units: The number of output units.
    :param input_shape: The shape of the input.
    :param normalize: Whether to normalize the input.
    :param kernel_regularizer: The kernel regularizer to use.
    :param dropout_rate: The dropout rate to use.

    :return: The model.
    """
    return ResNet(
        output_units,
        input_shape,
        (3, 8, 36, 3),
        "large",
        include_top=include_top,
        normalize=normalize,
        kernel_regularizer=kernel_regularizer,
        dropout_rate=dropout_rate,
    )


def write_summary(model: Model, file_path: str) -> None:
    """Write a summary of the model to a text file.

    :param model: The model to summarize.
    :param file_path: The path to the text file to write.
    """
    with open(file_path, "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))


if __name__ == "__main__":
    # Summarize models to test implementation.
    outputs = 1000
    input_shape = (224, 224, 3)
    normalize = True

    model = ResNet18(outputs, input_shape, normalize=normalize)
    write_summary(model, "resnet18.txt")

    model = ResNet34(outputs, input_shape, normalize=normalize)
    write_summary(model, "resnet34.txt")

    model = ResNet50(outputs, input_shape, normalize=normalize)
    write_summary(model, "resnet50.txt")

    model = ResNet101(outputs, input_shape, normalize=normalize)
    write_summary(model, "resnet101.txt")

    model = ResNet152(outputs, input_shape, normalize=normalize)
    write_summary(model, "resnet152.txt")
