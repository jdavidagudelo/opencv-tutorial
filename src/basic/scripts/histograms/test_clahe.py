from basic import load_images
from basic import histograms
import os

path = '{0}/../../images/image1.jpeg'.format(os.path.dirname(os.path.abspath(__file__)))
img = load_images.read_image(path)
new_img = histograms.create_clahe(img)
load_images.show_images([img, new_img])
