from datetime import datetime
from typing import List
import json
import argparse

class FoodMetadata:
    """ stores meta data on food images in COCO format"""
    def __init__(self, json_file_path = None):
        self.file_name = json_file_path

        # create a new json if none are supplied
        if self.file_name is None:
            self.coco = { 
                "info": {
                    "description": "Segmented Food Images from Google",
                    "version": "1.0",
                    "year": datetime.now().strftime('%Y'),
                    "contributors": "marcasty, pranav270-create",
                    "date_created": datetime.now().strftime('%Y-%m-%d')
                },
                "categories":[],
                "images": {},
                "annotations": {}
                }
        
        # create coco object from json file path
        else: 
            with open(self.file_name, 'r') as f:
                self.coco = json.load(f)

        # store the number of categories, images, annotations
        self.num_categories = len(self.coco["categories"])
        self.num_images = len(self.coco["images"])
        self.num_annotations= len(self.coco["annotations"])
        
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
    def get_num_images(self): return self.num_images

    def add_image_data(self, filename: str, search_query: str, width, height):
        """add an image to the coco json file
        key=image id, value=image metadata"""
        self.num_images += 1
        image_data = {"id": self.num_images,
                    "Google Search Query": search_query,
                    "width": width,
                    "height": height,
                    "filename": filename,
                    "date captured": datetime.now().strftime('%Y-%m-%d')
                    }
        self.coco["images"][self.num_images] = image_data

    # save the dict as a json file
    def export_coco(self, new_file_name = None, replace = False):
        if new_file_name is None:
            # do you want to replace old json file?
            if replace is True:
                assert self.file_name is not None, "this object was not created from a file so a file cannot be replaced"
                file_name = self.file_name
            else:
                current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                file_name = f"metadata_{current_datetime}.json"
        else: file_name = new_file_name

        with open(file_name, "w") as json_file:
            json.dump(self.coco, json_file, indent=4)
    
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
    # sam mask information is added directly in processing_pipeline.py ...hacky i know
    def add_dino_annot(self, classes, class_ids,  boxes, box_confidence):
        self.coco["annotations"][id]["num_objects"] = len(boxes)
        self.coco["annotations"][id]["classes"] = classes
        self.coco["annotations"][id]["class_ids"] = class_ids
        self.coco["annotations"][id]["xyxy_boxes"] = boxes
        self.coco["annotations"][id]["box_confidence"] = box_confidence
        self.coco["annotations"][id]["masks"] = []
        self.coco["annotations"][id]["mask_confidence"] = []

def parse_arguments():
    parser = argparse.ArgumentParser(description="Defines a metadata object in COCO format and scrapes Google for images of food.")
    parser.add_argument("--metadata_json", type=str, help="JSON file containing metadata in COCO format")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    # either opens supplied json or creates new coco file
    coco = FoodMetadata(args.metadata_json)
    print(coco.coco)

    # export metadata to json file
    coco.export_coco()