#!/usr/bin/python3
import os
import json
import csv
import fiftyone as fo
import fiftyone.zoo as foz
import argparse

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument(
    "--classes", type=list, default=["Person"], help=("List of classes to download")
)
parser.add_argument(
    "--num_samples", type=int, default=300, help="Number of samples to download."
)
parser.add_argument(
    "--download_folder",
    type=str,
    default="/root/fiftyone/",
    help="Folder where to download the images.",
)

args = parser.parse_args()

HOME = os.environ.get("HOME")
DATASET_NAME = "whatagan"
DATASET_DIR = f"{HOME}/fiftyone/{DATASET_NAME}"

# The splits to load
splits = ["train", "val", "persons"]

# Load the dataset, using tags to mark the samples in each split
dataset = fo.Dataset(DATASET_NAME)
for split in splits:
    dataset.add_dir(
        dataset_dir=DATASET_DIR,
        dataset_type=fo.types.YOLOv5Dataset,
        split=split,
        tags=split,
)

# View summary info about the dataset
print(dataset)

# Print the first few samples in the dataset
print(dataset.head())

'''
dataset = foz.load_zoo_dataset(
    "open-images-v6",
    split="validation",
    max_samples=args.num_samples,
    seed=51,
    label_types=["detections"],
    classes=args.classes,
    shuffle=True,
)

# The directory to which to write the exported dataset
export_dir = f"{DATASET_DIR}/test" 

# The type of dataset to export
# Any subclass of `fiftyone.types.Dataset` is supported
dataset_type=fo.types.YOLOv5Dataset

# Export the dataset
dataset.export(
    export_dir=export_dir,
    dataset_type=dataset_type,
)
'''

if __name__ == "__main__":
    session = fo.launch_app(dataset)
    # session.dataset = dataset
    session.wait()
