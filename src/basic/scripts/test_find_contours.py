from basic import load_images
from basic import contours

img = load_images.read_image('basic/images/image1.jpeg')
filtered, contours, hierarchy = contours.find_contours(img, 127, 255)
load_images.show_images([img, filtered])
