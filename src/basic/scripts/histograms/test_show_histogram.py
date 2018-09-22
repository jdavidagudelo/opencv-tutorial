from basic import histograms
from basic import load_images
import os

path = '{0}/../../images/image1.jpeg'.format(os.path.dirname(os.path.abspath(__file__)))

img = load_images.read_image(path)
histograms.show_histograms(img)