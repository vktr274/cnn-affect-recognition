from argparse import ArgumentParser
from distutils.dir_util import copy_tree
import logging
from pathlib import Path
from shutil import copy2
import pandas as pd
import os


def create_cli() -> ArgumentParser:
    """
    Creates a command line interface for the script.

    :return: ArgumentParser object.
    """
    cli = ArgumentParser(description="Relabels the dataset to the new labels")
    cli.add_argument(
        "path",
        type=str,
        help="Path to a directory that includes a train directory with the images in subdirectories named after the labels",
    )
    cli.add_argument(
        "--output-path", type=str, help="Path to an output directory", required=True
    )
    cli.add_argument(
        "--labels-csv",
        default="labels.csv",
        type=str,
        help="Name of the labels csv including file extension (default: 'labels.csv')",
    )
    return cli


def copy_to_output(from_path: str, to_path: str) -> None:
    """
    Copy files from one directory to another.

    :param from_path: Path to a directory with files.
    :param to_path: Path to a directory where files will be copied.
    """
    logging.info(f"Copying files from {from_path} to {to_path}")
    copy_tree(from_path, to_path)


def relabel(labels: pd.DataFrame, output_path: str) -> None:
    """
    Relabels the dataset to the new labels.

    :param labels: DataFrame with labels.
    """
    logging.info("Relabeling dataset to new labels")

    for _, row in labels.iterrows():
        path = row["pth"]
        old_label = path.split("/")[0]
        new_label = row["label"]
        if old_label == new_label:
            continue
        logging.info(f"Relabeling {old_label} to {new_label}")
        old_path = os.path.join(output_path, "train", path)
        copy2(old_path, os.path.join(output_path, "train", new_label))
        logging.info(f"Removing image with old label {old_path}")
        os.remove(old_path)


def main() -> None:
    cli = create_cli()
    args = cli.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not os.path.exists(args.path):
        logging.error(f"Path {args.path} does not exist")
        return
    if not os.path.exists(os.path.join(args.path, "train")):
        logging.error(f"Path {args.path} does not have train subdirectory")
    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    labels_path = os.path.join(args.path, args.labels_csv)
    if not os.path.exists(labels_path):
        logging.error(f"Directory {args.path} does not contain {args.labels_csv}")
        return

    copy_to_output(args.path, args.output_path)

    logging.info(f"Reading labels from {labels_path}")
    labels = pd.read_csv(labels_path, index_col=0)

    relabel(labels, args.output_path)


if __name__ == "__main__":
    main()
