from basic import load_images
from basic import morphological_transformations

img = load_images.read_image('basic/images/image1.jpeg')
filtered = morphological_transformations.dilate(img)
load_images.show_images([img, filtered])
