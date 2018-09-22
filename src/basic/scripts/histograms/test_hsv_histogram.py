from basic import load_images
from basic import histograms
import matplotlib.pyplot as plt
import os

path = '{0}/../../images/image1.jpeg'.format(os.path.dirname(os.path.abspath(__file__)))

img = load_images.read_image(path)
hist = histograms.get_hsv_histogram(img)
plt.imshow(hist, interpolation='nearest')
plt.show()

