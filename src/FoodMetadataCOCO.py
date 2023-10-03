from datetime import datetime
from pycocotools.coco import COCO
from typing import List
import json
import argparse
import numpy as np
import os
import torch


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return list(obj)
        return super(NpEncoder, self).default(obj)


class FoodMetadata(COCO):
    """ stores meta data on food images in COCO format"""
    def __init__(self, annotation_file=None):
        super().__init__(annotation_file)
        self.file_name = annotation_file
        if annotation_file is None:
            self.dataset = {
                'info': {
                "description": "Segmented Food Images",
                "version": "1.0",
                "year": datetime.now().strftime('%Y'),
                "contributors": "marcasty, pranav270-create",
                "date_created": datetime.now().strftime('%Y-%m-%d')
                },
                'categories':[],
                'images':[],
                'annotations':[]
            }
    # returns the number of 'categories' or meals found in dataset
    def get_num_categories(self): return self.num_categories

    # add new, unique foods to categories section
    def add_categories(self, new_foods: List[str]):
        old_food_set = set(self.coco["categories"])
        if isinstance(new_foods, list):
            new_foods_unique = [item for item in new_foods if item not in old_food_set]
            self.coco["categories"].extend(new_foods_unique)
        else:
            if new_foods not in old_food_set:
                self.coco["categories"].append(new_foods)

        # update number of categories
        self.num_categories = len(self.coco["categories"])

    # return number of images in the dataset
    def get_num_images(self):
        self.num_images = len(self.coco['images'])
        return self.num_images

    def add_image_data(self, filename: str, width, height, search_query: str = None):
        """add an image to the coco json file"""
        self.num_images += 1
        image_data = {"id": self.num_images,
                      "filename": filename,
                      "width": width,
                      "height": height,
                      "date captured": datetime.now().strftime('%Y-%m-%d')
                      }
        
        # append query if it was google searched
        if search_query is not None:
            image_data["Google Search Query"]: search_query

        self.dataset["images"].append(image_data) 


    # save the dict as a json file
    def export_coco(self, new_file_name=None, replace=False):
        if new_file_name is None:
            # do you want to replace old json file?
            if replace is True:
                assert self.file_name is not None, "this object was not created from a file so a file cannot be replaced"
                file_name = self.file_name
            else:
                current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                file_name = f"metadata_{current_datetime}.json"
        else:
            file_name = new_file_name

        with open(file_name, "w") as json_file:
            json.dump(self.dataset, json_file, indent=4, cls=NpEncoder)

    # return number of annotations
    def get_num_annotations(self): return self.num_annotations

    def add_annotation(self, id):
        """adds a blank entry to the annotations section of json"""
        new_annotation = {
            "id": id
        }
        self.coco["annotations"][id] = new_annotation

    def add_blip2_spacy_annot(self, id, text, words):
        """add blip2 and spacy results"""
        new_annotation = {
            "id": id,
            "blip2": text,
            "spacy": words
        }
        self.coco["annotations"][id] = new_annotation

    # adds dino annotations
    def add_dino_annot(self, id, classes, class_ids, boxes, box_confidence):
        self.coco["annotations"][id]["num_objects"] = len(boxes)
        self.coco["annotations"][id]["classes"] = classes
        self.coco["annotations"][id]["class_ids"] = class_ids
        self.coco["annotations"][id]["xyxy_boxes"] = boxes
        self.coco["annotations"][id]["box_confidence"] = box_confidence
        self.coco["annotations"][id]["masks"] = []
        self.coco["annotations"][id]["mask_confidence"] = []

    def add_sam_annot(self, id, arr_masks, arr_mask_score, directory):
        for i in range(len(arr_masks)):
            image_id = self.coco["annotations"][id]["id"]
            mask_id = f'{image_id}_{i}.pt'
            mask_filepath = os.path.join(directory, mask_id)
            torch.save(torch.Tensor(arr_masks[i]), mask_filepath)
            self.coco["annotations"][id]["masks"].append(mask_id)
            self.coco["annotations"][id]["mask_confidence"].append(arr_mask_score[i])


def parse_arguments():
    parser = argparse.ArgumentParser(description="Defines a metadata object in COCO format and scrapes Google for images of food.")
    parser.add_argument("--metadata_json", type=str, help="JSON file containing metadata in COCO format")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    # either opens supplied json or creates new coco file
    coco = FoodMetadata(args.metadata_json)
    print(coco.dataset['categories'])

    # export metadata to json file
    #coco.export_coco(new_file_name='new_format.json')