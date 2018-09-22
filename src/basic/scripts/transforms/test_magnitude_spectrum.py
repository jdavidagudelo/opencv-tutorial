from basic import load_images
from basic import transforms
import os

path = '{0}/../../images/image1.jpeg'.format(os.path.dirname(os.path.abspath(__file__)))
img = load_images.read_image(path)

filtered = transforms.get_magnitude_spectrum(img)
load_images.show_images([img, filtered])