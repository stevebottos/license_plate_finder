import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import os

def detect_img(yolo, img_path):
    print("Starting inference")
    for im in os.listdir(img_path):
        image = Image.open(img_path + im)

        print("Detecting")
        r_image = yolo.detect_image(image)
        r_image.show()
    yolo.close_session()

FLAGS = None


detect_img(YOLO(**{
    "model_path" : "trained_yolov3.h5",
    "anchors_path" : "model_data/yolo_anchors.txt",
    "classes_path" : "model_data/voc_classes.txt",
    "score": 0.25,
    "gpu_num": 0,
    "model_image_size": (416, 416)}),
    img_path="vehicles/test/")
