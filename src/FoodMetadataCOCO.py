from datetime import datetime
from collections import defaultdict
from pycocotools.coco import COCO
from typing import List
import json
import argparse
import numpy as np
import os
import torch

class FoodMetadata(COCO):
    """stores meta data on food images in COCO format"""

    def __init__(self, annotation_file=None, pred=False):
        super().__init__(annotation_file)
        self.file_name = annotation_file

        info = {
            "description": "Segmented Food Images",
            "version": "1.0",
            "year": datetime.now().strftime("%Y"),
            "contributors": "marcasty, pranav270-create",
            "date_created": datetime.now().strftime("%Y-%m-%d"),
        }

        # initialize new COCO JSON
        if annotation_file is None:
            self.dataset = {"info": info, "categories": [], "images": [], "annotations": []}
        # if making predictions on given data, remove given annotations
        else:
            if "detection_issues" not in self.dataset["info"]:
                self.dataset["info"]["detection_issues"] = {"failures": [], "detect_nonclass": []}
            if pred is True:
                self.anns = {}
                self.imgToAnns = {}
                self.dataset["annotations"] = []
                for img_id, _ in self.imgs.items():
                    self.imgToAnns[img_id] = []
                self.dataset["info"]["description"] = "Segmented Food Predictions"

    def get_num_categories(self):
        """how many categories (foods) are in this dataset?"""
        return len(self.cats)

    def add_categories(self, new_foods: List[str]):
        """
        -add new foods to categories
        -receives list of new foods; each element is words separated by spaces
        self.dataset["categories"] = [{
            'id': integer,
            'name': 'item-qualifier',
            'name_readable': 'Item, qualifier'
            'supercategory': 'food' }]
        """

        assert isinstance(new_foods, list), "please wrap new foods in list"

        existing_food_names = set(food["name"] for food in self.dataset["categories"])

        for food in new_foods:
            ingredients = food.split(" ")

            # prepare data to add to dicitonary
            if self.get_num_categories() > 0:
                id = self.dataset["categories"][-1]["id"] + 1
            else:
                id = 1
            name = "-".join(ingredients)
            name_readable = ", ".join(ingredient.capitalize() for ingredient in ingredients)

            if name not in existing_food_names:
                new_category = {"id": id, "name": name, "name_readable": name_readable, "supercategory": "food"}
                self.dataset["categories"].append(new_category)
                self.cats[id] = new_category
                self.catToImgs[id] = []

    def get_num_images(self):
        """return number of images in the dataset"""
        return len(self.imgs)

    def add_image_data(self, filename: str, width, height, cat_id, query: str = None):
        """add an image to the coco json file"""

        if self.get_num_images() > 0:
            id = max(self.imgs.keys()) + 1
        else:
            id = 1
        new_image = {
            "id": id,
            "category_id": cat_id,
            "filename": filename,
            "width": width,
            "height": height,
            "date captured": datetime.now().strftime("%Y-%m-%d"),
        }

        # append query if it was google searched
        if query is not None:
            new_image["Google Search Query"]: query

        self.imgs[id] = new_image
        self.catToImgs[cat_id].append(id)
        self.imgToAnns[id] = []

    def imgToCat(self, img_id):
        for cat, img_list in self.catToImgs.items():
            if img_id in img_list:
                return cat

    def get_num_annotations(self):
        """return number of annotations"""
        return len(self.anns)

    def next_ann_id(self):
        if self.get_num_annotations() == 0:
            return 1
        else:
            return max(self.anns.keys()) + 1

    def drop_ann(self, ann_ids: list):
        for ann_id in ann_ids:
            image_id = self.anns[ann_id]["image_id"]
            cat_id = self.anns[ann_id]["category_id"]
            self.anns.pop(ann_id)
            anns = []
            for ann in self.imgToAnns[image_id]:
                if ann["id"] != ann_id or ann["category_id"] != cat_id:
                    anns.append(ann)
            self.imgToAnns[image_id] = anns

    def update_imgToAnns(self, ann_id, image_id, key, value):
        for ann in self.imgToAnns[image_id]:
            if ann["id"] == ann_id:
                ann[key] = value

    def create_annot(self, image_id, cat_id):
        """initializes new id"""
        ann_id = self.next_ann_id()

        new_annotation = {
            "id": ann_id,
            "image_id": image_id,
            "category_id": cat_id,
        }
        self.anns[ann_id] = new_annotation
        self.imgToAnns[image_id].append(new_annotation)
        return ann_id

    def add_annot(self, ann_id, image_id, model, text):
        """add text annotation"""
        self.anns[ann_id][model] = text
        self.update_imgToAnns(ann_id, image_id, model, text)

    def add_dino_annot(self, ann_id, image_id, detections):
        """add DINO annotations"""

        self.add_annot(ann_id, image_id, "num_objects", len(detections["bbox"]))
        self.add_annot(ann_id, image_id, "classes", detections["classes"])

        for i in range(len(detections["bbox"])):
            if "bbox" in self.anns[ann_id]:
                new_annotation = self.anns[ann_id].copy()
                ann_id = self.next_ann_id()
                new_annotation["id"] = ann_id
                self.anns[ann_id] = new_annotation
                self.imgToAnns[image_id].append(new_annotation)
            self.add_annot(ann_id, image_id, "class_id", detections["class_id"][i])
            self.add_annot(ann_id, image_id, "bbox", detections["bbox"][i])
            self.add_annot(ann_id, image_id, "box_confidence", round(float(detections["box_confidence"][i]), 4))

    def add_sam_annot(self, ann_id, image_id, mask, mask_confidence, mask_dir):
        if mask_dir:
            mask_id = f"{image_id}_{ann_id}.pt"
            mask_filepath = os.path.join(mask_dir, mask_id)
            torch.save(torch.Tensor(mask), mask_filepath)
            self.add_annot(ann_id, image_id, "mask", mask_id)
        else:
            self.add_annot(ann_id, image_id, "mask", mask)
        self.add_annot(ann_id, image_id, "mask_confidence", round(float(mask_confidence), 4))

    # save the dict as a json file
    def export_coco(self, new_file_name=None):
        if new_file_name is None:
            current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file_name = f"metadata_{current_datetime}.json"
        else:
            file_name = new_file_name

        self.dataset["categories"] = list(self.cats.values())
        self.dataset["images"] = list(self.imgs.values())
        self.dataset["annotations"] = list(self.anns.values())

        print(f"saving data as {file_name}")
        with open(file_name, "w") as json_file:
            json.dump(self.dataset, json_file, indent=4)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Defines a metadata object in COCO format and scrapes Google for images of food."
    )
    parser.add_argument("--metadata_json", type=str, help="JSON file containing metadata in COCO format")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # either opens supplied json or creates new coco file
    coco = FoodMetadata(args.metadata_json)

    # if you want to keep the annotations of the json file:
    val_data = FoodMetadata("public_validation_set_2.1_blip_spacy.json")

    # if you want to make new predictions on the data, set pred=True
    prediction = FoodMetadata("public_validation_set_release_2.1.json", pred=True)
    print(prediction.dataset["images"][0])
    print(prediction.dataset["annotations"])
    # export metadata to json file
    # val_data.export_coco(new_file_name='data.json')

    """
    there are multiple annotations per image
    there are multiple images per category
    some images have multiple categories
    .catToImgs and imgToAnns are dicts in FoodMetadata class to keep track of this
    """
