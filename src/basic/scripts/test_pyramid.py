from basic import load_images
from basic import pyramids

img = load_images.read_image('basic/images/image1.jpeg')

filtered = pyramids.get_pyramid_up(img)
load_images.show_images([img, filtered])
