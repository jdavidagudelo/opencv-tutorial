from basic import load_images
from basic import filtering_operations

img = load_images.read_image('basic/images/image1.jpeg')
noisy = filtering_operations.add_noise_to_image(img, 0.05)
filtered = filtering_operations.bilateral_filtering(noisy, 5, 5, 5)
load_images.show_images([noisy, filtered])