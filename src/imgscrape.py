from PIL import Image
import argparse
from icrawler.builtin import GoogleImageCrawler
from FoodMetadataCOCO import FoodMetadata
import os

# scrape google for new images, save relevant metadata according to coco format
def crawl_google_images(metadata, new_foods, save_dir, quantity):
    if quantity is None: quantity = 10

    # add truly new foods to list of categories 
    num_cat_old = metadata.get_num_categories()
    metadata.add_categories(new_foods)

    # execute the crawl across the entire list of new foods 
    for food in metadata.coco["categories"][num_cat_old:]:
        print(f'FOOD: {food}')
        google_crawler = GoogleImageCrawler(storage = {'root_dir': save_dir})

        #crawl
        google_crawler.crawl(keyword=food, max_num=quantity)

        # make sure the query does not contain characters that are forbidden in filenames
        forbidden_chars = ' <>:"/\\|?*_'  # Include underscore for later convenience
        clean_filename = ''.join(['-' if char in forbidden_chars else char for char in food])

        for idx in range(0, quantity):
            # get new image
            try: 
                img_path = f"{save_dir}/{idx+1:06d}.jpg"
                img = Image.open(img_path)
                img_type = 'jpg'
            except:
                img_path = f"{save_dir}/{idx+1:06d}.png"
                img = Image.open(img_path)
                img_type = 'png'
            width, height, = img.size
            img.close()

            # form new name based on query and index
            img_name = f'{clean_filename}_{idx+1:05}.{img_type}' 
            
            # add meta data to JSON file: filename, query, width, height
            metadata.add_image_data(img_name, food, width, height)

            # create new path
            new_img_path = f"{save_dir}/{img_name}"
            os.rename(img_path, new_img_path)
            
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

    # read in list of new foods
    with open(args.new_foods_text, "r") as f:
        new_foods = [line.strip() for line in f.readlines()]

    # crawl 
    crawl_google_images(coco, new_foods[:2], args.img_dir, args.num_examples)
    
    # export metadata to json file
    coco.export_coco()
