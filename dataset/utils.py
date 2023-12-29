#!/usr/bin/python3 
import shutil
import json
import os
import csv
import tqdm
import yaml
import argparse

HOME = os.environ.get("HOME")

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument(
    "--dataset", type=str, default="open-images-v6", help=("Name of dataset to load"),
)

parser.add_argument(
    "--read_classes", type=bool, default=False, help="Create class .json file"
)

parser.add_argument(
    "--add_classes", type=bool, default=False, help="Add new class .json file"
)

args = parser.parse_args()

DATASET_DIR = f"{HOME}/fiftyone/{args.dataset}"

if args.read_classes:
    # Create class dictionary
    classDict = {}
    with open(f"{DATASET_DIR}/validation/metadata/classes.csv", "r") as f:
        rows = csv.reader(f, delimiter=",")
        for row in rows:
            classDict[row[0]] = row[1]

        f.close()

    with open(f"{DATASET_DIR}/validation/metadata/classes.json", "w") as f:
        f.write(json.dumps(classDict))
        f.close()

if args.add_class:
    # for file in os.listdir(args.dataset):

    # Read new class names in new yaml file
    new_class_list = []
    new_class_dict = {}
    with open(f"{DATASET_DIR}/dataset.yaml", "r") as f:
        new_config = yaml.safe_load(f)
        new_class_list = new_config["names"]
        f.close()

    # Read existing class names in original yaml file
    old_classes = []
    with open(f"{old_config}/dataset.yaml", "r") as f:
        old_config = yaml.safe_load(f)["names"]
        old_class_list = yaml.safe_load(f)["names"]
        f.close()

    # Check new yaml file for class name
    new_class_amount = len(old_class_list)
    if new_class_list in old_class_list:
        new_class_dict[str(new_class_list)] = str(old_class_list.index(new_class_list))
    else:
        new_class_dict[str(new_class_amount)] = str(new_class_amount)
        new_class_amount += 1





# If class name not in yaml add to yaml and
# increment training data by one
