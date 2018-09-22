from basic import load_images
from basic import pyramids
import os


path = '{0}/../../images/image1.jpeg'.format(os.path.dirname(os.path.abspath(__file__)))
img = load_images.read_image(path)

filtered = pyramids.get_pyramid_up(img)
load_images.show_images([img, filtered])
