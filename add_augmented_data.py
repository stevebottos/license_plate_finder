import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import cv2
import os
import pandas as pd
import ast
import numpy as np
import time
import shutil
pd.set_option("display.max_rows", None,
"display.max_columns", None,
'display.expand_frame_repr', False,
'display.max_colwidth', -1)

"""
We'll use image augmentation to synthetically add to the dataset
"""

batches = [1, 2]

def aug_pts_to_dict(bbs_aug):
    boxes = {}
    dict_idx = 1
    for i in range(len(bbs_aug)):
        boxes[dict_idx] = {}
        boxes[dict_idx]["xmin"] = int(round(bbs_aug[i].x1,0))
        boxes[dict_idx]["xmax"] = int(round(bbs_aug[i].x2,0))
        boxes[dict_idx]["ymin"] = int(round(bbs_aug[i].y1,0))
        boxes[dict_idx]["ymax"] = int(round(bbs_aug[i].y2,0))
        boxes[dict_idx]["region_name"] = 1
        dict_idx += 1
    return boxes

for batch in batches:
    images = pd.read_csv("vehicles/batch" + str(batch) + "_formatted_og.csv")
    ims_og_path = "vehicles/batch" + str(batch) + "_og/"
    ims_path = "vehicles/batch" + str(batch) + "/"

    # For new data
    filename_dat = []
    boxes_dat = []

    for _, im in images.iterrows():
        f = im["fname"]

        shutil.copy(ims_og_path + f, "vehicles/batch" + str(batch) + "/" + f)

        i = cv2.imread(ims_og_path + f)

        attributes = ast.literal_eval(im["roi"])
        bbs = []
        for key in attributes:
            rect = attributes[key]
            x1 = int(rect["xmin"])
            x2 = int(rect["xmax"])
            y1 = int(rect["ymin"])
            y2 = int(rect["ymax"])

            bbs.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2))

        bbs = BoundingBoxesOnImage(bbs, i.shape)

        """
        STEPS
        1) Identiy a transformation
        2) Convert its resultant bounding boxes to the proper format
        3) Save the image
        4) Save the fname and boxes as a result to be appended to the final csv

        ... In non-shift cases the bounding boxes will be the same as otherwise, but for consistency we'll just keep doing the transforms
        """
        f = f.split(".")[0]
        print(f)
        # - - - - WITH BRIGHTNESS SHIFTED - - - - #
        seq = iaa.Multiply((0.4, 0.5))
        darkened, bbs_aug = seq(image=i, bounding_boxes=bbs)
        darkened_bbs = aug_pts_to_dict(bbs_aug)
        filename_dat += [f + "_darkened" + ".jpg"]
        boxes_dat += [darkened_bbs]
        cv2.imwrite(ims_path + f + "_darkened" + ".jpg", darkened)

        seq = iaa.Rot90(1)
        rot90, rot90_bbs = seq(image=i, bounding_boxes=bbs)
        rot90_bbs = aug_pts_to_dict(rot90_bbs)
        filename_dat += [f + "_darkened_rot90"+ ".jpg"]
        boxes_dat += [rot90_bbs]
        cv2.imwrite(ims_path + f + "_darkened_rot90"+ ".jpg", rot90)

        seq = iaa.Rot90(2)
        rot180, rot180_bbs = seq(image=i, bounding_boxes=bbs)
        rot180_bbs = aug_pts_to_dict(rot180_bbs)
        filename_dat += [f + "_darkened_rot180" + ".jpg"]
        boxes_dat += [rot180_bbs]
        cv2.imwrite(ims_path + f + "_darkened_rot180" + ".jpg", rot180)

        seq = iaa.Rot90(3)
        rot270, rot270_bbs = seq(image=i, bounding_boxes=bbs)
        rot270_bbs = aug_pts_to_dict(rot270_bbs)
        filename_dat += [f + "_darkened_rot270" + ".jpg"]
        boxes_dat += [rot270_bbs]
        cv2.imwrite(ims_path + f + "_darkened_rot270" + ".jpg", rot270)

        # # - - - - GREYSCALE - - - - #
        seq = iaa.color.ChangeColorspace("GRAY")
        grey, grey_bbs = seq(image=i, bounding_boxes=bbs)
        grey_bbs = aug_pts_to_dict(grey_bbs)
        filename_dat += [f + "_grey" + ".jpg"]
        boxes_dat += [grey_bbs]
        cv2.imwrite(ims_path + f + "_grey" + ".jpg", grey)

        seq = iaa.Rot90(1)
        rot90, rot90_bbs = seq(image=grey, bounding_boxes=bbs)
        rot90_bbs = aug_pts_to_dict(rot90_bbs)
        filename_dat += [f + "_grey_rot90"+ ".jpg"]
        boxes_dat += [rot90_bbs]
        cv2.imwrite(ims_path + f + "_grey_rot90"+ ".jpg", rot90)

        seq = iaa.Rot90(2)
        rot180, rot180_bbs = seq(image=grey, bounding_boxes=bbs)
        rot180_bbs = aug_pts_to_dict(rot180_bbs)
        filename_dat += [f + "_grey_rot180" + ".jpg"]
        boxes_dat += [rot180_bbs]
        cv2.imwrite(ims_path + f + "_grey_rot180" + ".jpg", rot180)

        seq = iaa.Rot90(3)
        rot270, rot270_bbs = seq(image=grey, bounding_boxes=bbs)
        rot270_bbs = aug_pts_to_dict(rot270_bbs)
        filename_dat += [f + "_grey_rot270" + ".jpg"]
        boxes_dat += [rot270_bbs]
        cv2.imwrite(ims_path + f + "_grey_rot270" + ".jpg", rot270)

    row_data = list(zip(filename_dat, boxes_dat))
    formatted = pd.DataFrame(row_data, columns = ["fname", "roi"])

    formatted = pd.concat([images, formatted]).reset_index().drop(columns="Unnamed: 0")
    formatted.to_csv("vehicles/batch" + str(batch) + "_formatted.csv")
    print(formatted.shape)
