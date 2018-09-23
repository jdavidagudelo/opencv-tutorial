from basic import load_images
from basic import corners
import os

path = '{0}/../../images/chess.jpg'.format(os.path.dirname(os.path.abspath(__file__)))
img = load_images.read_image(path)
filtered = corners.get_fast_features(img, 150)
load_images.show_images([img, filtered])
