from basic import load_images
from basic import basic_operations
import os
import numpy as np

path = '{0}/../../images/image1.jpeg'.format(os.path.dirname(os.path.abspath(__file__)))
img = load_images.read_image(path)

pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

transformed = basic_operations.get_perspective_transform(img, pts1, pts2, 300, 300)
load_images.show_image(transformed)
