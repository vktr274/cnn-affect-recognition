import numpy as np
import pandas as pd
from argparse import ArgumentParser
import logging
import os
import shutil
from pathlib import Path
import albumentations as A
import cv2


def calculate_class_ratio(class_df: pd.DataFrame, largest_class_count: int) -> float:
    """
    Calculate ratio with respect to the largest class.

    :param class_df: DataFrame with labels and paths for a single class.
    :param largest_class: Largest class size.
    """
    return len(class_df) / largest_class_count


def balance_class(
    df: pd.DataFrame,
    data_path: str,
    subset: str,
    ratio: float,
    label_col: str,
    path_col: str,
    seed=None,
) -> pd.DataFrame:
    """
    Balance class by augmenting it.

    :param df: DataFrame with labels and paths for a single class.
    :param ratio: Ratio for class.
    :param label_col: Name of label column.
    :param path_col: Name of path column.
    """
    pipeline = A.Compose(
        [
            A.HorizontalFlip(),
            A.RandomBrightnessContrast(
                always_apply=True, contrast_limit=0.2, brightness_limit=0.2
            ),
            A.Rotate(always_apply=True, limit=20, border_mode=cv2.BORDER_REPLICATE),
        ]
    )
    augment_count = int(ratio * len(df))
    new_df = pd.DataFrame(columns=[label_col, path_col])
    df_sample = df.sample(n=augment_count, random_state=seed)
    for _, row in df_sample.iterrows():
        path = row[path_col]
        label = row[label_col]
        filename = os.path.basename(path)
        image = cv2.imread(os.path.join(data_path, subset, path))
        augmented = pipeline(image=image)
        augmented_image = augmented["image"]
        new_path = os.path.join(label, f"augmented_{filename}")
        new_df = pd.concat(
            [new_df, {label_col: label, path_col: new_path}], ignore_index=True
        )
        cv2.imwrite(os.path.join(data_path, subset, new_path), augmented_image)
    return new_df


def copy_test(df: pd.DataFrame, data_path: str, label_col: str, path_col: str) -> None:
    """
    Copy test files to test subdirectory.

    :param df: DataFrame with labels and paths.
    :param data_path: Path to data directory.
    :param label_col: Name of label column.
    :param path_col: Name of path column.
    """
    logging.info(f"Copying {len(df)} test files from train to test subdirectory")
    for _, row in df.iterrows():
        path = row[path_col]
        label = row[label_col]
        train_image_path = os.path.join(data_path, "train", path)
        test_path = os.path.join(data_path, "test", label)
        Path(test_path).mkdir(parents=True, exist_ok=True)
        shutil.copy2(train_image_path, test_path)


def delete_copied(df: pd.DataFrame, data_path: str, path_col: str) -> None:
    """
    Delete copied test files from train subdirectory.

    :param df: DataFrame with labels and paths.
    :param data_path: Path to data directory.
    :param path_col: Name of path column.
    """
    logging.info(f"Deleting {len(df)} copied test files from train subdirectory")
    for _, row in df.iterrows():
        path = row[path_col]
        train_path = os.path.join(data_path, "train", path)
        os.remove(train_path)


def split_data(
    df: pd.DataFrame,
    data_path: str,
    train_split=0.8,
    balance=False,
    label_col="label",
    path_col="path",
    seed=None,
) -> None:
    """
    Split data into training and test sets.

    :param df: DataFrame with labels and paths.
    :param train_split: Train split ratio.
    :param balance: Balance classes in training set.
    :param label_col: Name of label column.
    :param path_col: Name of path column.
    :param seed: Random seed.
    """
    classes = df[label_col].unique()
    logging.info(f"Classes: {classes}")

    counts = df[label_col].value_counts()
    largest_class_size = counts.max()
    largest_train_class_size = np.ceil(largest_class_size * train_split)
    logging.info(f"Largest train class size: {largest_train_class_size}")

    train_df = pd.DataFrame(columns=[label_col, path_col])
    test_df = pd.DataFrame(columns=[label_col, path_col])

    for c in classes:
        class_paths = df[df[label_col] == c][path_col]
        train_paths = class_paths.sample(frac=train_split, random_state=seed)
        test_paths = class_paths.drop(train_paths.index)

        logging.info(f"Class {c}: {len(train_paths)} train samples")
        logging.info(f"Class {c}: {len(test_paths)} test samples")

        train_class_df = pd.DataFrame({path_col: train_paths, label_col: c})
        test_class_df = pd.DataFrame({path_col: test_paths, label_col: c})
        copy_test(test_class_df, data_path, label_col, path_col)
        delete_copied(test_class_df, data_path, path_col)

        if balance:
            ratio_train = calculate_class_ratio(
                train_class_df, largest_train_class_size
            )
            ratio_test = calculate_class_ratio(
                test_class_df, largest_class_size - largest_train_class_size
            )
            logging.info(f"Class {c}: ratio_train = {ratio_train}")
            logging.info(f"Class {c}: ratio_test = {ratio_test}")
            if ratio_train < 1:
                train_class_df = balance_class(
                    train_class_df,
                    data_path,
                    "train",
                    ratio_train,
                    label_col,
                    path_col,
                    seed,
                )
            if ratio_test < 1:
                test_class_df = balance_class(
                    test_class_df,
                    data_path,
                    "test",
                    ratio_test,
                    label_col,
                    path_col,
                    seed,
                )
            logging.info(f"Class {c}: augmented to {len(train_class_df)} train samples")
            logging.info(f"Class {c}: augmented to {len(test_class_df)} test samples")
        train_df = pd.concat([train_df, train_class_df], ignore_index=True)
        test_df = pd.concat([test_df, test_class_df], ignore_index=True)

    logging.info(f"Total: {len(train_df)} train samples")
    logging.info(f"Total: {len(test_df)} test samples")
    train_df.to_csv(os.path.join(data_path, "train.csv"), index=False)
    test_df.to_csv(os.path.join(data_path, "test.csv"), index=False)


if __name__ == "__main__":
    parser = ArgumentParser(description="Split data into training and test sets.")
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Train split ratio (default: 0.8)",
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Balance classes in training set (default: False)",
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default="labels",
        help="Name of csv file (default: labels)",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="label",
        help="Name of label column (default: label)",
    )
    parser.add_argument(
        "--path-col",
        type=str,
        default="path",
        help="Name of path column (default: path)",
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to a directory with train subdirectory and labels.csv file.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed (default: None)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

    if not os.path.exists(args.path):
        logging.error(f"Path {args.path} does not exist.")

    train_path = os.path.join(args.path, "train")
    if not os.path.exists(train_path):
        logging.error(f"Path {args.path} does not include train subdirectory.")

    labels_path = os.path.join(args.path, args.csv_name + ".csv")
    if not os.path.exists(labels_path):
        logging.error(f"Path {args.path} does not include labels.csv file.")

    df = pd.read_csv(labels_path)
    split_data(
        df=df,
        data_path=args.path,
        train_split=args.train_split,
        balance=args.balance,
        label_col=args.label_col,
        path_col=args.path_col,
        seed=args.seed,
    )
