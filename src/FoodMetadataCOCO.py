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

        # initialize new COCO JSON
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
            else: id = 1
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
            id = self.dataset['images'][-1]["id"] + 1
        else: id = 1
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

        self.dataset["images"].append(new_image) 
        self.imgs[id] = new_image
        self.catToImgs[cat_id].append(id)
        self.imgToAnns[id] = []

    def get_num_annotations(self): 
        """return number of annotations"""
        return len(self.anns)

    def add_annotation(self, image_id):
        """initializes new id"""
        if self.get_num_annotations() > 0:
            id = self.dataset['annotations'][-1]["id"] + 1
        else: id = 1
        
        new_annotation = {
            "id": id,
            "image_id": image_id,
            "category_id": self.imgs[image_id]["category_id"],
        }
        self.dataset["annotations"].append(new_annotation)
        self.anns[id] = new_annotation
        self.imgToAnns[image_id].append(id)

    def add_blip2_spacy_annot(self, image_id, text, words):
        """add blip2 and spacy results"""

        # initialize annotation
        self.add_annotation(image_id)

        self.dataset["annotations"][-1]["blip2"] = text
        self.dataset["annotations"][-1]["spacy"] = words

        id = self.dataset["annotations"][-1]["id"]
        self.anns[id]["blip2"] = text
        self.anns[id]["spacy"] = words


    # adds dino annotations
    def add_dino_annot(self, img_id, ann_id, classes, class_ids, boxes, box_confidence):
        dino_ann_ids = []
        
        # craft a new annotation
        new_annotation = self.anns[ann_id]
        new_annotation["num_objects"] = len(boxes)
        new_annotation["classes"] = classes
        for i in range(0, len(boxes)):
            new_annotation["class_ids"] = class_ids[i]
            new_annotation["xyxy_boxes"] = boxes[i]
            new_annotation["box_confidence"] = box_confidence[i]
            
            # if not first box saved to image, add new annotation
            if 'xyxy_boxes' in self.anns[ann_id]:
                # update id info
                id = self.dataset['annotations'][-1]['id'] + 1
                new_annotation["id"] = id
                dino_ann_ids.append(id)
                # add new annotation
                self.dataset['annotations'].append(new_annotation)
                self.anns[id] = new_annotation
                self.imgToAnns[img_id].append(id)

            # if this is first box saved to image, update first instance of annotation
            else:
                dino_ann_ids.append(ann_id)
                self.dataset["annotations"][self.id_to_idx(ann_id)] = new_annotation
                self.anns[ann_id] = new_annotation
        
        return dino_ann_ids

    def add_sam_annot(self, ann_ids, arr_masks, arr_mask_score, directory):
        for i, ann_id in enumerate(ann_ids):
            image_id = self.anns[ann_id]["image_id"]
            mask_id = f'{image_id}_{ann_id}.pt'
            mask_filepath = os.path.join(directory, mask_id)
            torch.save(torch.Tensor(arr_masks[i]), mask_filepath)
            self.dataset['annotations'][self.id_to_idx(ann_id)]['masks'] = mask_id
            self.dataset['annotations'][self.id_to_idx(ann_id)]['mask_confidence'] = arr_mask_score[i]
            self.anns['masks'] = mask_id
            self.anns['mask_confidence'] = arr_mask_score[i]

    def id_to_idx(self,id):
        return [d for d in self.dataset['annotations'] if d.get('id') == id][0]

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