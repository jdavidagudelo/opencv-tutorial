from basic import load_images
from basic import transforms
import os

path = '{0}/../../images/image3.png'.format(os.path.dirname(os.path.abspath(__file__)))
img = load_images.read_image(path)

filtered = transforms.get_hough_circles(img)
load_images.show_images([img, filtered])
