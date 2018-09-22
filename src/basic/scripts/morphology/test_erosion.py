from basic import load_images
from basic import morphological_transformations
import os

path = '{0}/../../images/image1.jpeg'.format(os.path.dirname(os.path.abspath(__file__)))
img = load_images.read_image(path)
filtered = morphological_transformations.erode(img)
load_images.show_images([img, filtered])
