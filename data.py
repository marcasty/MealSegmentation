from datetime import datetime
import os
from PIL import Image
# Machine Learning Core
import json
import argparse
from icrawler.builtin import GoogleImageCrawler

class COCO_MetaData:
    """ stores meta data on images in COCO format"""
    def __init__(self, json_file_path = None):
        # create a new json if none are supplied
        if json_file_path is None:
            self.coco = { 
                "info": {
                    "description": "MealSegmentation Dataset from Google Images",
                    "version": "1.0",
                    "year": 2023,
                    "contributor": "marcasty, pranav270-create",
                    "date_created": "2023-09-15"
                },
                "categories":[],
                "images": [],
                "annotations": []
                }
        
        # create coco object from json file
        else: 
            with open(json_file_path, 'r') as f:
                self.coco = json.load(f)

        self.num_categories = len(self.coco["categories"])
        self.num_images = len(self.coco["images"])
        self.num_annotations= len(self.coco["annotations"])

    # returns the number of 'categories' or meals found in dataset
    def get_num_categories(self): return self.num_categories

    # add new categories to scrape from the web
    def add_categories(self, new_foods):
        # create a set of current foods for efficient comparisons
        old_food_set = set(self.coco["categories"])
        if isinstance(new_foods, list):
            new_foods_unique = [item for item in new_foods if item not in old_food_set]
            self.coco["categories"].extend(new_foods_unique)
        else:
            if new_foods not in old_food_set:
                self.coco["categories"].append(new_foods)
        self.num_categories = len(self.coco["categories"])

    # returns number of images in the dataset
    def get_num_images(self): return self.num_images

    # adds image meta data to json
    def add_image_data(self, filename, search_query, width, height):
        image_data = {
            "id": self.num_images + 1,
            "Google Search Query": search_query,
            "width": width,
            "height": height,
            "filename": filename,
            "date captured": datetime.now().strftime('%Y-%m-%d')
        }
        self.coco["images"].append(image_data)
        self.num_images += 1

    # save the dict as a json file
    def export_coco(self, replace = False):
        # do you wish to replace the existing json file?
        if replace is False:
            current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file_name = f"data_{current_datetime}.json"
        else: file_name = "data.json"

        with open(file_name, "w") as json_file:
            json.dump(self.coco, json_file)
    
    def get_num_annotations(self): return self.num_annotations

    def add_annotation(self):
        return 1

# scrape google for new images, save relevant metadata according to coco format
def crawl_google_images(metadata, new_foods, save_dir, quantity):
    # set up crawler
    google_Crawler = GoogleImageCrawler(storage = {'root_dir': save_dir})

    # save a list of images already in the directory
    old_images = os.listdir(save_dir)

    # add truly new foods to list of categories 
    num_cat_old = metadata.get_num_categories()
    metadata.add_categories(new_foods)
    num_cat_new = metadata.get_num_categories()

    # execute the crawl across the entire list of new foods 
    for i in range(num_cat_old + 1, num_cat_new):
        if quantity is None:
            quantity = 1
        google_Crawler.crawl(keyword=metadata.coco["categories"][i], max_num=quantity)

        # list of newly scraped filenames
        new_images = list(set(os.listdir(save_dir)) - set(old_images))
        
        # make sure the query does not contain characters that are forbidden in filenames
        forbidden_chars = ' <>:"/\\|?*_'  # Include underscore for later convenience
        clean_filename = ''.join(['-' if char in forbidden_chars else char for char in metadata.coco["categories"][i]])
        
        for idx, image_name in enumerate(new_images):
            # save old path
            image_path = f"{save_dir}/{image_name}"
                                                  
            # get image width/height
            image = Image.open(image_path)
            width, height, = image.size
            image.close()

            # form new name based on query and index
            image_num = f"{idx+1:05}"
            new_image_name = f'{clean_filename}_{image_num}.jpg'  # if image_name.endswith('.jpg') else f'{query}_{idx+1}.png'
            
            # add meta data to JSON file: filename, query, width, height
            metadata.add_image_data(new_image_name, metadata.coco["categories"][i], width, height)

            # create new path
            new_image_path = f"{save_dir}/{new_image_name}"
            # rename image to new path
            os.rename(image_path, new_image_path)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Defines a metadata object in COCO format and scrapes Google for images of food.")
    parser.add_argument("img_dir", type=str, help="Directory to save images")
    parser.add_argument("--metadata_json", type=str, help="JSON file containing metadata in COCO format")
    parser.add_argument("new_foods_text", type=str, help="text file containing new foods to scrape")
    parser.add_argument("--num_examples", type=int, help="number of examples per food you wish to scrape")
    parser.add_argument("--query", type=str, help="a single query to scrape")
    return parser.parse_args()

# must include:
#   save directory
#   textfile containing prospective new foods
if __name__ == '__main__':
    args = parse_arguments()

    # either opens supplied json or creates new coco file
    coco = COCO_MetaData(args.metadata_json)
    with open(args.new_foods_text, "r") as f:
        new_foods = [line.strip() for line in f.readlines()]

    crawl_google_images(coco, new_foods, args.img_dir, args.num_examples)

    # export metadata to json file
    coco.export_coco()
