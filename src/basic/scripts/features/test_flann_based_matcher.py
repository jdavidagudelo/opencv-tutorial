from basic import load_images
from basic import corners
import os

path1 = '{0}/../../images/objects.jpg'.format(os.path.dirname(os.path.abspath(__file__)))
path2 = '{0}/../../images/query.jpg'.format(os.path.dirname(os.path.abspath(__file__)))
img1 = load_images.read_image(path1)
img2 = load_images.read_image(path2)
filtered = corners.fast_nn_based_matcher(img1, img2)
load_images.show_images([filtered])
