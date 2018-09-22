from basic import load_images
from basic import basic_operations

img = load_images.read_image('basic/images/image1.jpeg')
translated = basic_operations.translate_image(img, 100, 100, img.shape[0], img.shape[1])
load_images.show_image(translated)