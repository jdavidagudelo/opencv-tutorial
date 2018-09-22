from basic import load_images
from basic import basic_operations
import numpy as np

img = load_images.read_image('basic/images/image1.jpeg')

pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

transformed = basic_operations.get_affine_transform(img, pts1, pts2, img.shape[0], img.shape[1])

load_images.show_images([img, transformed])
