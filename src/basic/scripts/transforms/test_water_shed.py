from basic import load_images
from basic import transforms
import os

path = '{0}/../../images/water_coins.jpg'.format(os.path.dirname(os.path.abspath(__file__)))
img = load_images.read_image(path)

filtered = transforms.water_shed_images(img)
load_images.show_images([img, filtered])
