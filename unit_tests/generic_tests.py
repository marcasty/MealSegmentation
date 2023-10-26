import os
import sys
import argparse

current_file_dir = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.abspath(os.path.join(current_file_dir, os.pardir))
sys.path.append(os.path.join(parent_directory, "src"))
from FoodMetadataCOCO import FoodMetadata


def check_missing_annotations(coco: FoodMetadata) -> None:
    ann_ids = sorted(coco.anns.keys())
    expected_id = 1
    fail = 0
    for ann_id in ann_ids:
        if ann_id == expected_id:
            expected_id += 1
        else:
            print(f"Missing annotation: {expected_id}")
            expected_id += 1
            fail = 1
    if fail == 1:
        raise AssertionError("Your pipeline has removed annotations from the JSON file")
    else:
        print(f"Test Passed! Your pipeline did not remove any keys")


def parse_arguments():
    parser = argparse.ArgumentParser(description="test COCO json files")
    parser.add_argument(
        "--json_file", type=str, help="JSON file containing metadata in COCO format"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    metadata = FoodMetadata(args.json_file)
    check_missing_annotations(metadata)
