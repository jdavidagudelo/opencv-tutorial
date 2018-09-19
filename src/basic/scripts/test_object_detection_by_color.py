import numpy as np
from basic import object_tracking_by_color
from basic import load_images


img = load_images.read_image('basic/images/image3.png')
high_color = np.array([130,255,255])
low_color = np.array([110, 50, 50])
img, mask, res = object_tracking_by_color.detect_object_by_hsv_color(img, low_color, high_color)
load_images.show_images([img, mask, res])
