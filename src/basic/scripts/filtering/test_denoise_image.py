from basic import load_images
from basic import filtering_operations
import os

path = '{0}/../../images/image1.jpeg'.format(os.path.dirname(os.path.abspath(__file__)))
img = load_images.read_image(path)
noisy = filtering_operations.add_noise_to_image(img, 0.05)
filtered = filtering_operations.de_noise_image(noisy, 10, 10, 7, 21)
load_images.show_images([noisy, filtered])
