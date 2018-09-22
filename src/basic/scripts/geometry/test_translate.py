from basic import load_images
from basic import basic_operations
import os

path = '{0}/../../images/image1.jpeg'.format(os.path.dirname(os.path.abspath(__file__)))
img = load_images.read_image(path)
translated = basic_operations.translate_image(img, 100, 100, img.shape[0], img.shape[1])
load_images.show_image(translated)