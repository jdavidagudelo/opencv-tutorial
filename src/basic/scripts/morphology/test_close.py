from basic import load_images
from basic import morphological_transformations, filtering_operations
import os

path = '{0}/../../images/image1.jpeg'.format(os.path.dirname(os.path.abspath(__file__)))
img = load_images.read_image(path)
noisy = filtering_operations.add_noise_to_image(img, 0.05)

filtered = morphological_transformations.image_close(img)
load_images.show_images([noisy, filtered])