from basic import load_images
from basic import morphological_transformations, filtering_operations

img = load_images.read_image('basic/images/image1.jpeg')
noisy = filtering_operations.add_noise_to_image(img, 0.05)

filtered = morphological_transformations.image_open(img)
load_images.show_images([noisy, filtered])