from basic import load_images
from basic import contours
import os

path = '{0}/../../images/image1.jpeg'.format(os.path.dirname(os.path.abspath(__file__)))
img = load_images.read_image(path)
filtered, c, hierarchy = contours.find_contours(img, 127, 255)
new_img = contours.draw_contours(img, c, -1, 0, 255, 0, 3)
load_images.show_images([img, new_img])
