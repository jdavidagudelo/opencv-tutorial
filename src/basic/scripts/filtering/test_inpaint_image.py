from basic import load_images
from basic import filtering_operations
from basic import basic_operations
import os
import cv2

path = '{0}/../../images/inpaint_result.jpg'.format(os.path.dirname(os.path.abspath(__file__)))
path1 = '{0}/../../images/inpaint_mask.jpg'.format(os.path.dirname(os.path.abspath(__file__)))
img = load_images.read_image(path)
mask = load_images.read_image(path1, 0)

mask = basic_operations.make_same_size(mask, img)
filtered = filtering_operations.in_paint_image(img, mask)
load_images.show_images([img, filtered])
