from PIL import Image
import argparse
from icrawler.builtin import GoogleImageCrawler
from FoodMetadataCOCO import FoodMetadata
import os
import json

# scrape google for new images, save relevant metadata according to coco format
def crawl_google_images(metadata, new_foods, save_dir, quantity, json_path = None):
    """
    new_foods are new google search queries
    they get added to categories if they're new"""
    if quantity is None: quantity = 10

    # add truly new foods to list of categories 
    num_cat_old = metadata.get_num_categories()
    metadata.add_categories(new_foods)

    # execute the crawl across the entire list of new foods 
    for food in metadata.dataset["categories"][num_cat_old:]:
        food_name = food["name"]
        print(f'FOOD: {food_name}')

        # save each category in their own subdirectory
        sub_save_dir = os.path.join(save_dir, food_name)
        os.makedirs(sub_save_dir, exist_ok = True)
                
        # initialize crawler
        google_crawler = GoogleImageCrawler(storage = {'root_dir': sub_save_dir})

        # replace '-' with ' ' for google query purposes
        query = food_name.replace('-', ' ')
        google_crawler.crawl(keyword=query, max_num=quantity)

        for idx in range(0, quantity):
            # get new image; if the file is not .jpg or png, continue
            try: 
                img_path = f"{sub_save_dir}/{idx+1:06d}.jpg"
                img = Image.open(img_path)
                img_type = 'jpg'
            except:
                try:
                    img_path = f"{sub_save_dir}/{idx+1:06d}.png"
                    img = Image.open(img_path)
                    img_type = 'png'
                except:
                    break
            width, height, = img.size
            img.close()

            # form new name based on query and index
            img_name = f'/{food_name}/{food_name}_{idx+1:05}.{img_type}' 
            
            # add meta data to JSON file: filename, query, width, height
            metadata.add_image_data(img_name, width, height, food["id"], query)

            # create new path
            new_img_path = f"{save_dir}{img_name}"
            os.rename(img_path, new_img_path)
        
        # save json as you go, if desired
        if json_path is not None:
            with open(json_path, "w") as json_file:
                json.dump(metadata.dataset, json_file, indent=4)
            
def parse_arguments():
    parser = argparse.ArgumentParser(description="Defines a metadata object in COCO format and scrapes Google for images of food.")
    parser.add_argument("img_dir", type=str, help="Directory to save images")
    parser.add_argument("--metadata_json", type=str, help="JSON file containing metadata in COCO format")
    #parser.add_argument("new_foods_text", type=str, help="text file containing new foods to scrape")
    parser.add_argument("--num_examples", type=int, help="number of examples per food you wish to scrape")
    parser.add_argument("--query", type=str, help="a single query to scrape")
    return parser.parse_args()

# must include:
#   save directory
#   textfile containing prospective new foods
if __name__ == '__main__':
    args = parse_arguments()

    # either opens supplied json or creates new coco file
    coco = FoodMetadata(args.metadata_json)

    # read in list of new foods
    #with open(args.new_foods_text, "r") as f:
    #    new_foods = [line.strip() for line in f.readlines()]

    # crawl 
    crawl_google_images(coco, ['bacon and eggs', 'burger and fries', 'apple and banana'], args.img_dir, 2)
    
    # export metadata to json file
    coco.export_coco('data.json')
    
    print('*' * 20)
    print(f'imgs {coco.imgs}')
    print('*' * 20)
    print(f'cats {coco.cats}')
    print('*' * 20)
    print(f'catToImgs {coco.catToImgs}')
