from basic import load_images
from basic import histograms
import matplotlib.pyplot as plt

img = load_images.read_image('basic/images/image1.jpeg')
hist = histograms.get_hsv_histogram(img)
plt.imshow(hist, interpolation='nearest')
plt.show()

