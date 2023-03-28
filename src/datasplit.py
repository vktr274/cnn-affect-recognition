import numpy as np
import pandas as pd
from argparse import ArgumentParser
import logging
import os
import shutil
from pathlib import Path
import albumentations as A
import cv2


def calculate_class_multiplier(
    class_df: pd.DataFrame,
    largest_class_count: int,
    global_multiplier=1.0,
) -> float:
    """
    Calculate multiplier with respect to the largest class.

    :param class_df: DataFrame with labels and paths for a single class.
    :param largest_class: Largest class size.
    :return: Multiplier.
    """
    return global_multiplier * largest_class_count / len(class_df)


def balance_class(
    df: pd.DataFrame,
    data_path: str,
    subset: str,
    ratio: float,
    label_col: str,
    filename_col: str,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Balance class by augmenting it.

    :param df: DataFrame with labels and paths for a single class.
    :param data_path: Path to data directory.
    :param subset: Subset name.
    :param ratio: Ratio for class.
    :param label_col: Name of label column.
    :param filename_col: Name of filename column.
    :param seed: Random seed.
    :return: Augmented DataFrame.
    """
    pipeline = A.Compose(
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
    augment_count = int(np.ceil(ratio * len(df))) - len(df)
    df_sample = df.sample(n=augment_count, random_state=seed, replace=True)
    row: pd.Series[str]
    for _, row in df_sample.iterrows():
        filename = row[filename_col]
        label = row[label_col]
        image = cv2.imread(os.path.join(data_path, subset, label, filename))
        augmented = pipeline(image=image)
        augmented_image = augmented["image"]
        random_filename = f"aug_{np.random.randint(0, 1_000_000)}{filename}"
        df = pd.concat(
            [df, pd.DataFrame({label_col: [label], filename_col: [random_filename]})],
            ignore_index=True,
        )
        cv2.imwrite(
            os.path.join(data_path, subset, label, random_filename), augmented_image
        )
    return df


def copy_test(
    df: pd.DataFrame, data_path: str, label_col: str, filename_col: str
) -> None:
    """
    Copy test files to test subdirectory.

    :param df: DataFrame with labels and paths.
    :param data_path: Path to data directory.
    :param label_col: Name of label column.
    :param filename_col: Name of filename column.
    """
    row: pd.Series[str]
    for _, row in df.iterrows():
        filename = row[filename_col]
        label = row[label_col]
        train_image_path = os.path.join(data_path, "train", label, filename)
        test_path = os.path.join(data_path, "test", label)
        Path(test_path).mkdir(parents=True, exist_ok=True)
        shutil.copy2(train_image_path, test_path)


def delete_copied(
    df: pd.DataFrame, data_path: str, label_col: str, filename_col: str
) -> None:
    """
    Delete copied test files from train subdirectory.

    :param df: DataFrame with labels and paths.
    :param data_path: Path to data directory.
    :param label_col: Name of label column.
    :param filename_col: Name of filename column.
    """
    row: pd.Series[str]
    for _, row in df.iterrows():
        label = row[label_col]
        filename = row[filename_col]
        train_path = os.path.join(data_path, "train", label, filename)
        os.remove(train_path)


def split_data(
    df: pd.DataFrame,
    data_path: str,
    train_split=0.8,
    balance=False,
    label_col="label",
    filename_col="filename",
    global_multiplier=1.0,
    seed: int | None = None,
) -> None:
    """
    Split data into training and test sets and optionally balance classes.

    :param df: DataFrame with labels and filenames.
    :param data_path: Path to data directory.
    :param train_split: Train split ratio.
    :param balance: Whether to balance classes in training and test set.
    :param label_col: Name of label column.
    :param filename_col: Name of filename column.
    :param global_multiplier: Global multiplier for class size.
    :param seed: Random seed.
    """
    classes = df[label_col].unique()
    logging.info(f"Classes: {classes}")
    logging.info(f"Global multiplier: {global_multiplier}")

    counts = df[label_col].value_counts()
    largest_class_size = counts.max()
    largest_train_class_size = int(np.ceil(largest_class_size * train_split))
    logging.info(f"Largest train class size: {largest_train_class_size}")

    train_df = pd.DataFrame(columns=[label_col, filename_col])
    test_df = pd.DataFrame(columns=[label_col, filename_col])

    c: str
    for c in classes:
        df_c = df[df[label_col] == c]
        train_class_df = df_c.sample(frac=train_split, random_state=seed)
        test_class_df = df_c.drop(train_class_df.index).astype(str)

        logging.info(f"Class {c}: {len(train_class_df)} train samples")
        logging.info(f"Class {c}: {len(test_class_df)} test samples")

        logging.info(
            f"Copying {len(test_class_df)} test files from train to test subdirectory"
        )
        copy_test(test_class_df, data_path, label_col, filename_col)
        logging.info(
            f"Deleting {len(test_class_df)} copied test files from train subdirectory"
        )
        delete_copied(test_class_df, data_path, label_col, filename_col)

        if balance:
            multiplier_train = calculate_class_multiplier(
                train_class_df, largest_train_class_size, global_multiplier
            )
            multiplier_test = calculate_class_multiplier(
                test_class_df,
                largest_class_size - largest_train_class_size,
                global_multiplier,
            )
            logging.info(f"Class {c}: multiplier_train = {multiplier_train}")
            logging.info(f"Class {c}: multiplier_test = {multiplier_test}")
            if multiplier_train > 1.0:
                train_class_df = balance_class(
                    train_class_df,
                    data_path,
                    "train",
                    multiplier_train,
                    label_col,
                    filename_col,
                    seed,
                )
                logging.info(
                    f"Class {c}: augmented to {len(train_class_df)} train samples"
                )
            if multiplier_test > 1.0:
                test_class_df = balance_class(
                    test_class_df,
                    data_path,
                    "test",
                    multiplier_test,
                    label_col,
                    filename_col,
                    seed,
                )
                logging.info(
                    f"Class {c}: augmented to {len(test_class_df)} test samples"
                )
        train_df = pd.concat([train_df, train_class_df], ignore_index=True)
        test_df = pd.concat([test_df, test_class_df], ignore_index=True)

    logging.info(f"Total: {len(train_df)} train samples")
    logging.info(f"Total: {len(test_df)} test samples")
    train_df.to_csv(os.path.join(data_path, "train.csv"), index=False)
    test_df.to_csv(os.path.join(data_path, "test.csv"), index=False)


def load_dataframe(train_path: str, label_col: str, filename_col: str) -> pd.DataFrame:
    """
    Load DataFrame with labels and filenames.

    :param train_path: Path to train subdirectory.
    :param label_col: Name of label column.
    :param filename_col: Name of filename column.
    :return: DataFrame with labels and filenames.
    """
    df = pd.DataFrame(columns=[label_col, filename_col])
    for label in os.listdir(train_path):
        label_path = os.path.join(train_path, label)
        for filename in os.listdir(label_path):
            df = pd.concat(
                [df, pd.DataFrame({label_col: [label], filename_col: [filename]})],
                ignore_index=True,
            )
    return df


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Split data into training and test sets and optinally augment or balance classes."
    )
    parser.add_argument(
        "--train-split",
        type=lambda x: float(x)
        if float(x) > 0.5
        else parser.error(
            "Train split ratio must be a floating point number greater than 0.5"
        ),
        default=0.8,
        help="Train split ratio (default: 0.8)",
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Balance classes in training set (default: False)",
    )
    parser.add_argument(
        "--seed",
        type=lambda x: int(x)
        if int(x) > 0
        else parser.error("Seed must be an integer greater than 0"),
        default=None,
        help="Random seed (default: None)",
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to a directory with train subdirectory.",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="label",
        help="Name of label column you want to be created in the CSV files (default: label)",
    )
    parser.add_argument(
        "--filename-col",
        type=str,
        default="filename",
        help="Name of filename column you want to be created in the CSV files (default: filename)",
    )
    parser.add_argument(
        "--global-multiplier",
        type=lambda x: float(x)
        if float(x) >= 1.0
        else parser.error(
            "Multiplier must be a floating point number greater than or equal to 1.0"
        ),
        default=1.0,
        help="Global multiplier for all classes (default: 1.0)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

    if not os.path.exists(args.path):
        logging.error(f"Path {args.path} does not exist.")

    train_path = os.path.join(args.path, "train")
    if not os.path.exists(train_path):
        logging.error(f"Path {args.path} does not include train subdirectory.")

    logging.info(f"Creating DataFrame of labels and filenames from {train_path}")
    df = load_dataframe(train_path, args.label_col, args.filename_col)

    split_data(
        df=df,
        data_path=args.path,
        train_split=args.train_split,
        balance=args.balance,
        label_col=args.label_col,
        filename_col=args.filename_col,
        global_multiplier=args.global_multiplier,
        seed=args.seed,
    )
