import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import cv2
import os

im_path = "vehicles/batch1"
flist = os.listdir(im_path)



for f in flist:
    print(f)





#
# ia.seed(1)
#
# image = ia.quokka(size=(256, 256))
# bbs = BoundingBoxesOnImage([
#     BoundingBox(x1=65, y1=100, x2=200, y2=150),
#     BoundingBox(x1=150, y1=80, x2=200, y2=130)
# ], shape=image.shape)
#
# seq = iaa.Sequential([
#     iaa.Multiply((1.2, 1.5)), # change brightness, doesn't affect BBs
#     iaa.Affine(
#         translate_px={"x": 40, "y": 60},
#         scale=(0.5, 0.7)
#     ) # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
# ])
#
# # Augment BBs and images.
# image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
#
# # print coordinates before/after augmentation (see below)
# # use .x1_int, .y_int, ... to get integer coordinates
# for i in range(len(bbs.bounding_boxes)):
#     before = bbs.bounding_boxes[i]
#     after = bbs_aug.bounding_boxes[i]
#     print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
#         i,
#         before.x1, before.y1, before.x2, before.y2,
#         after.x1, after.y1, after.x2, after.y2)
#     )
#
# # image with BBs before/after augmentation (shown below)
# image_before = bbs.draw_on_image(image, size=2)
# image_after = bbs_aug.draw_on_image(image_aug, size=2, color=[0, 0, 255])
