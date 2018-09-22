from basic import load_images
from basic import basic_operations

img = load_images.read_image('basic/images/image1.jpeg')
scaled = basic_operations.resize_image_with_scale(img, 2, 2)
load_images.show_image(scaled)
