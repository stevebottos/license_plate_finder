import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import cv2
import os
import pandas as pd
import ast
import numpy as np

"""
We'll use image augmentation to synthetically add to the dataset
"""

def aug_pts_to_dict(bbs_aug):
    boxes = {}
    dict_idx = 1
    for i in range(len(bbs_aug)):
        boxes[dict_idx] = {}
        boxes[dict_idx]["xmin"] = int(round(bbs_aug[i].x1,0))
        boxes[dict_idx]["xmax"] = int(round(bbs_aug[i].x2,0))
        boxes[dict_idx]["ymin"] = int(round(bbs_aug[i].y1,0))
        boxes[dict_idx]["ymax"] = int(round(bbs_aug[i].y2,0))
        boxes[dict_idx]["region_name"] = "plate"
        dict_idx += 1
    return boxes



images = pd.read_csv("vehicles/batch1_formatted.csv")
ims_path = "vehicles/batch1/"

# For new data
filename_dat = []
boxes_dat = []

for _, im in images.iterrows():
    i = cv2.imread(ims_path + im["fname"])

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

    seq = iaa.Multiply((0.4, 0.5))
    image_aug, bbs_aug = seq(image=i, bounding_boxes=bbs)
    dict_pts = aug_pts_to_dict(bbs_aug)
    print(dict_pts)





    # image with BBs before/after augmentation (shown below)
    # image_before = bbs.draw_on_image(i, size=2)
    # image_after = bbs_aug.draw_on_image(image_aug, size=2, color=[0, 0, 255])
    #
    # cv2.imshow("b", image_before)
    # cv2.imshow("a", image_after)
    # cv2.waitKey(1000)
    # cv2.destroyAllWindows()
