from basic import load_images
from basic import filtering_operations

img = load_images.read_image('basic/images/image1.jpeg')
filtered = filtering_operations.add_noise_to_image(img, 0.05)
load_images.show_images([img, filtered])