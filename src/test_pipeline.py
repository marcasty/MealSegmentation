import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np
from FoodMetadataCOCO import FoodMetadata
import supervision as sv

# Let us make a function that adds masks and boxes and classes to the existing dataset
def add_masks_and_labels(img_dir, data_file, box_thresh=0.35, text_thresh=0.25):
    # read in metadata
    metadata = FoodMetadata(data_file)

    for id, annot in metadata['images'].items():
        SOURCE_IMAGE_PATH = f"{img_dir}{annot['filename']}"

        image = cv2.imread(SOURCE_IMAGE_PATH)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        id = metadata.coco["annotations"][id]['id']

        # IMAGE ID
        blip2 = metadata.coco["annotations"][id]['blip2']
        spacy = metadata.coco["annotations"][id]['spacy']

        # DINO STUFF
        num_objects = metadata.coco["annotations"][id]["num_objects"]
        classes = metadata.coco["annotations"][id]["classes"]
        class_ids = metadata.coco["annotations"][id]["class_ids"]
        boxes = metadata.coco["annotations"][id]["xyxy_boxes"]
        box_confidence = metadata.coco["annotations"][id]["box_confidence"]

        # SAM STUFF
        mask = metadata.metadata.coco["annotations"][id]["masks"]
        scores = metadata.metadata.coco["annotations"][id]["mask_confidence"]
        # save_filename = 

        plot_sample(image_rgb, boxes, box_confidence, classes, class_ids, save_filename)  



import matplotlib.patches as patches
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt


def img_show(img, ax = None, figsize=(7,11)):
  fig, ax = plt.subplots(figsize=figsize)
  #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  ax.xaxis.tick_top()
  ax.imshow(img)
  return ax

def get_bb(box):
  x = box[0]
  y = box[1]
  w = box[2] - box[0]
  h = box[3] - box[1]
  return [x,y,w,h]

def draw_box(img, ax, bb):
  patch = ax.add_patch(patches.Rectangle((bb[0],bb[1]), bb[2], bb[3], fill=False, edgecolor='red', lw=2))

def draw_outline(obj):
  obj.set_path_effects([patheffects.Stroke(linewidth=4,  foreground='black'), patheffects.Normal()])

def draw_text(ax, bb, txt, disp):
  text = ax.text(bb[0],(bb[1]-disp),txt,verticalalignment='top',color='white',fontsize=10,weight='bold')
  draw_outline(text)

def plot_sample(img, bboxes, box_confidence, classes, class_ids, filename, ax=None, figsize=(7,11)):
    ax = img_show(img, ax=ax, figsize=figsize)
    for i in range(len(bboxes)):
        bb = get_bb(bboxes[i])
        draw_box(img,ax,bb)
        label = f'{str(classes[class_ids[i]])}, {box_confidence[class_ids[i]]:.2f}'
        draw_text(ax, bb, label, img.shape[0]*0.05)
    plt.savefig(filename)

def multiplot(image, boxes, classes, class_ids, figsize=(20, 5)):
    num_objects = len(boxes)
    fig, ax = plt.subplots(1, num_objects + 1, figsize=figsize)
    plt.subplots_adjust(wspace=0.1, hspace=0)
    fig.tight_layout()
    plot_sample(image, boxes, classes, class_ids, ax=ax[0])
    for i, axs in enumerate(ax):
        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        label = f'{str(classes[class_ids[i]])}, {box_confidence[class_ids[i]]:.2f}'
        plot_sample(img, [boxes[i]], classes, [class_ids[i]], ax=ax[i + 1])