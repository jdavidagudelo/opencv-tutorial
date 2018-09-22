from basic import load_images
from basic import gradient_operations

img = load_images.read_image('basic/images/image1.jpeg')

filtered = gradient_operations.sobel(img, 1, 0, 5)
load_images.show_images([img, filtered])
