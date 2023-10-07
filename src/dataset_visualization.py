import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as patheffects
import cv2
import numpy as np
from FoodMetadataCOCO import FoodMetadata
import torch
import sys
import os
sys.path.append('../src')
sys.path.append('../')


# Let us make a function that adds masks and boxes and classes to the existing dataset
def add_masks_and_labels(img_dir, mask_dir, data_file):
    # read in metadata
    metadata = FoodMetadata(data_file)

    count = 0
    for id, annot in metadata.coco['images'].items():
        count += 1
        if count > 4:
            break
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
        mask = metadata.coco["annotations"][id]["masks"]
        scores = metadata.coco["annotations"][id]["mask_confidence"]

        save_filename_box = f'../data/test_box_{id}.png'
        save_filename_multibox = f'../data/test_multibox_{id}.png'
        save_filename_mask = f'../data/test_mask_{id}.png'
        save_filename_multimask = f'../data/test_multimask_{id}.png'

        plot_boxes(image_rgb, boxes, box_confidence, classes, class_ids, save_filename_box)
        multiplot_boxes(image_rgb, boxes, box_confidence, classes, class_ids, save_filename_multibox)
        plot_masks(image_rgb, mask, scores, classes, class_ids, mask_dir, save_filename_mask)
        multiplot_masks(image_rgb, mask, scores, classes, class_ids, mask_dir, save_filename_multimask)


def img_show(img, ax=None, figsize=(7, 11)):
    fig, ax = plt.subplots(figsize=figsize)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ax.xaxis.tick_top()
    ax.imshow(img)
    return ax


def get_bb(box):
    x = box[0]
    y = box[1]
    w = box[2] - box[0]
    h = box[3] - box[1]
    return [x, y, w, h]


def draw_box(img, ax, bb):
    patch = ax.add_patch(patches.Rectangle((bb[0],bb[1]), bb[2], bb[3], fill=False, edgecolor='red', lw=2))


def draw_outline(obj):
    obj.set_path_effects([patheffects.Stroke(linewidth=4,  foreground='black'), patheffects.Normal()])


def draw_text(ax, bb, txt, disp):
    text = ax.text(bb[0], (bb[1]-disp), txt, verticalalignment='top', color='white', fontsize=10, weight='bold')
    draw_outline(text)


def plot_boxes(img, bboxes, box_confidence, classes, class_ids, filename, ax=None, figsize=(7, 11)):
    ax = img_show(img, ax=ax, figsize=figsize)
    for i in range(len(bboxes)):
        bb = get_bb(bboxes[i])
        draw_box(img, ax, bb)
        label = f'{str(classes[class_ids[i]])}, {box_confidence[i]:.2f}'
        draw_text(ax, bb, label, img.shape[0]*0.05)
    plt.savefig(filename)


def plot_masks(img, masks_list, mask_confidence, classes, class_ids, load_path, filename, figsize=(20, 5)):
    masked_image = np.zeros_like(img)
    plt.figure(figsize=figsize)
    legend = ''
    for i in range(len(masks_list)):
        food_mask = torch.load(os.path.join(load_path, masks_list[i])).int().detach().numpy()
        masked_image[food_mask == 1] = img[food_mask == 1]
        label = f'{str(classes[class_ids[i]])}, {mask_confidence[i]:.2f}'
        legend += (label)
        legend += '    '
    plt.title(legend)
    plt.imshow(masked_image)
    plt.savefig(filename)


def multiplot_masks(img, masks_list, mask_confidence, classes, class_ids, load_path, filename, figsize=(20, 5)):
    num_objects = len(masks_list)
    if num_objects > 1:
        fig, ax = plt.subplots(1, num_objects, figsize=figsize)
        plt.subplots_adjust(wspace=0.1, hspace=0)
        fig.tight_layout()
        for i, axs in enumerate(ax):
            food_mask = torch.load(os.path.join(load_path, masks_list[i])).int().detach().numpy()
            masked_image = np.zeros_like(img)
            masked_image[food_mask == 1] = img[food_mask == 1]
            label = f'{str(classes[class_ids[i]])}, {mask_confidence[i]:.2f}'
            axs.set_title(label)
            axs.imshow(masked_image)
        plt.savefig(filename)
    else:
        plot_masks(img, masks_list, mask_confidence, classes, class_ids, load_path, filename, figsize=(20, 5))


def multiplot_boxes(img, boxes, box_confidence, classes, class_ids, filename, figsize=(20, 5)):
    num_objects = len(boxes)
    if num_objects > 1:
        fig, ax = plt.subplots(1, num_objects, figsize=figsize)
        plt.subplots_adjust(wspace=0.1, hspace=0)
        fig.tight_layout()
        for i, axs in enumerate(ax):
            axs.imshow(img)
            bb = get_bb(boxes[i])
            draw_box(img, axs, bb)
            label = f'{str(classes[class_ids[i]])}, {box_confidence[i]:.2f}'
            draw_text(axs, bb, label, img.shape[0]*0.05)
        plt.savefig(filename)
    else:
        plot_boxes(img, boxes, box_confidence, classes, class_ids, filename, figsize=(20, 5))


if __name__ == '__main__':
    img_dir = "../images"
    mask_dir = "../masks"
    data_file = "../google_food101_10k_dedup_keywords_masks.json"
    add_masks_and_labels(img_dir, mask_dir, data_file)
