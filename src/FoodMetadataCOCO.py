from datetime import datetime
from collections import defaultdict
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
    def __init__(self, annotation_file=None, pred=False):
        super().__init__(annotation_file)
        self.file_name = annotation_file

        info = {
            "description": "Segmented Food Images",
            "version": "1.0",
            "year": datetime.now().strftime('%Y'),
            "contributors": "marcasty, pranav270-create",
            "date_created": datetime.now().strftime('%Y-%m-%d')
            }

        # initialize new COCO JSON
        if annotation_file is None:
            self.dataset = {
                'info': info,
                'categories': [],
                'images': [],
                'annotations': []
            }
        # if making predictions on given data, remove given annotations 
        else:
            if pred is True:
                self.anns = {}
                self.imgToAnns = {}
                self.dataset['annotations'] = []
                for img_id, _ in self.imgs.items():
                    self.imgToAnns[img_id] = []
                info["description"] = "Segmented Food Predictions"
                self.dataset['info'] = info

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

        existing_food_names = set(food['name'] for food in self.dataset["categories"])

        for food in new_foods:
            ingredients = food.split(' ')

            # prepare data to add to dicitonary
            if self.get_num_categories() > 0:
                id = self.dataset['categories'][-1]['id'] + 1
            else:
                id = 1
            name = '-'.join(ingredients)
            name_readable = ', '.join(ingredient.capitalize() for ingredient in ingredients)

            if name not in existing_food_names:
                new_category = {
                    'id': id,
                    'name': name,
                    'name_readable': name_readable,
                    'supercategory': 'food'
                    }
                self.dataset['categories'].append(new_category)
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
        new_image = {"id": id,
                     "category_id": cat_id,
                      "filename": filename,
                      "width": width,
                      "height": height,
                      "date captured": datetime.now().strftime('%Y-%m-%d')
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
        if self.get_num_annotations() == 0: return 1
        else: return max(self.anns.keys()) + 1
    
    def update_imgToAnns(self, ann_id, image_id, key, value):
        for ann in self.imgToAnns[image_id]:
            if ann['id'] == ann_id:
                ann[key] = value

    def add_annotation(self, image_id, cat_id):
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

    def add_blip2_annot(self, ann_id, image_id, text):
        """add blip2 results"""

        self.anns[ann_id]["blip2"] = text
        self.update_imgToAnns(ann_id, image_id, "blip2", text)

    def add_spacy_annot(self, ann_id, image_id, words):
        """add spacy results"""

        self.anns[ann_id]["spacy"] = words
        self.update_imgToAnns(ann_id, image_id, "spacy", words)


    def add_class_from_embd(self, ann_id, mod_classes, classes):
        """add class name nearest to blip/spacy output"""
        
        image_id = self.anns[ann_id]["image_id"]

        self.anns[ann_id]["mod_class_from_embd"] = mod_classes
        self.update_imgToAnns(ann_id, image_id, "mod_class_from_embd", mod_classes)
        self.anns[ann_id]["class_from_embd"] = classes
        self.update_imgToAnns(ann_id, image_id, "class_from_embd", classes)

    # adds dino annotations
    def add_dino_annot(self, img_id, ann_id, classes, class_ids, boxes, box_confidence):
        dino_ann_ids = []

        # craft a new annotation
        new_annotation = self.anns[ann_id]
        new_annotation["num_objects"] = len(boxes)
        new_annotation["classes"] = classes
        for i in range(0, len(boxes)):
            new_annotation["class_ids"] = class_ids[i]
            new_annotation["bbox"] = boxes[i]
            new_annotation["box_confidence"] = box_confidence[i]

            if 'bbox' not in self.anns[ann_id]:
                self.anns[ann_id] = new_annotation
                self.imgToAnns[img_id] = new_annotation

            # if this is not the first box saved to an image, add new annotation
            else:
                id = self.next_ann_id()
                new_annotation["id"] = id
                self.anns[id] = new_annotation
                self.imgToAnns[img_id].append(new_annotation)

            dino_ann_ids.append(ann_id)

        return dino_ann_ids

    def add_sam_annot(self, ann_ids, arr_masks, arr_mask_score, directory):
        for i, ann_id in enumerate(ann_ids):
            image_id = self.anns[ann_id]["image_id"]
            mask_id = f'{image_id}_{ann_id}.pt'
            mask_filepath = os.path.join(directory, mask_id)
            torch.save(torch.Tensor(arr_masks[i]), mask_filepath)
            self.anns[ann_id]['masks'] = mask_id
            self.anns[ann_id]['mask_confidence'] = arr_mask_score[i]

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
        
        # inverse the index
        categories, images, annotations = [], [], []

        for _, cat in self.cats.items():
            categories.append(cat)
        
        for _, img in self.imgs.items():
            images.append(img)
        
        for _, ann in self.anns.items():
            annotations.append(ann)
        
        self.dataset['categories'] = categories
        self.dataset['images'] = images
        self.dataset['annotations'] = annotations

        with open(file_name, "w") as json_file:
            json.dump(self.dataset, json_file, indent=4, cls=NpEncoder)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Defines a metadata object in COCO format and scrapes Google for images of food.")
    parser.add_argument("--metadata_json", type=str, help="JSON file containing metadata in COCO format")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    # either opens supplied json or creates new coco file
    coco = FoodMetadata(args.metadata_json)

    # if you want to keep the annotations of the json file: 
    val_data = FoodMetadata('public_validation_set_2.1_blip_spacy.json')

    # if you want to make new predictions on the data, set pred=True
    prediction = FoodMetadata('public_validation_set_release_2.1.json', pred=True)
    print(prediction.dataset['images'][0])
    print(prediction.dataset['annotations'])
    # export metadata to json file
    #val_data.export_coco(new_file_name='data.json')

    """
    there are multiple annotations per image
    there are multiple images per category
    .catToImgs and imgToAnns are dicts in FoodMetadata class to keep track of this
    """