from basic import load_images
from basic import gradient_operations
import os

path = '{0}/../../images/image1.jpeg'.format(os.path.dirname(os.path.abspath(__file__)))
img = load_images.read_image(path)

filtered = gradient_operations.sobel(img, 0, 1, 5)
load_images.show_images([img, filtered])
