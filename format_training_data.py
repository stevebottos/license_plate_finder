import pandas as pd
import ast
import os

batches = [1,2]
with open('train.txt', 'w') as trainfile:
    for batch in batches:

        images = pd.read_csv("vehicles/batch" + str(batch) + "_formatted.csv")
        img_path = "vehicles/batch"+ str(batch) + "/"


        for _, im in images.iterrows():
            write_str = ""

            f = im["fname"]
            f_nospaces = f.replace(" ", "")
            p = img_path + "batch" + str(batch) + f_nospaces
            os.rename(img_path + f, p)


            write_str += p + " "

            attributes = ast.literal_eval(im["roi"])
            bbs = []
            for key in attributes:
                rect = attributes[key]
                x1 = int(rect["xmin"])
                x2 = int(rect["xmax"])
                y1 = int(rect["ymin"])
                y2 = int(rect["ymax"])

                write_str += str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ',' + "0 "
                print(write_str)
            trainfile.write(write_str + "\n")
