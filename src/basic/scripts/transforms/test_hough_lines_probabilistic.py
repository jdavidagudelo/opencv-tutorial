from basic import load_images
from basic import transforms
import os

path = '{0}/../../images/sudoku.jpg'.format(os.path.dirname(os.path.abspath(__file__)))
img = load_images.read_image(path)

filtered = transforms.get_hough_lines_probabilistic(img, 50, 250, 3, threshold=200)
load_images.show_images([img, filtered])
