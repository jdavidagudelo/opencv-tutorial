from basic import load_images
from basic import basic_operations

img = load_images.read_image('basic/images/image1.jpeg')
rotated = basic_operations.rotate_image(img, 1, 90, img.shape[0], img.shape[1])
load_images.show_image(rotated)
