from basic import load_images
from basic import basic_operations
import os

path = '{0}/../../images/image1.jpeg'.format(os.path.dirname(os.path.abspath(__file__)))
img = load_images.read_image(path)
scaled = basic_operations.resize_image_with_scale(img, 2, 2)
load_images.show_image(scaled)
