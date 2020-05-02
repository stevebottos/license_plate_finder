import os
import pandas as pd
import ast
pd.set_option("display.max_rows", None,
"display.max_columns", None,
'display.expand_frame_repr', False,
'display.max_colwidth', -1)

"""
This script just formats the vgg csv in a way that's easier to work with
"""


raw_dat = pd.read_csv("vehicles/batch1_via_region_data.csv")
im_path = "vehicles/batch1_og"
flist = os.listdir(im_path)

filename_dat = []
boxes_dat = []
for f in flist:
    boxes = {}
    image_info = raw_dat.loc[raw_dat["filename"] == f]

    dict_idx = 1
    for _, row in image_info.iterrows():
        attributes = ast.literal_eval(row["region_shape_attributes"])
        boxes[dict_idx] = {}
        boxes[dict_idx]["xmin"] = attributes["x"]
        boxes[dict_idx]["ymin"] = attributes["y"]
        boxes[dict_idx]["xmax"] = attributes["x"] + attributes["width"]
        boxes[dict_idx]["ymax"] = attributes["y"] + attributes["height"]
        boxes[dict_idx]["region_name"] = 0

        dict_idx += 1

    filename_dat += [f]
    boxes_dat += [boxes]

row_data = list(zip(filename_dat, boxes_dat))
formatted = pd.DataFrame(row_data, columns = ["fname", "roi"])
formatted.to_csv("vehicles/batch1_formatted_og.csv")
