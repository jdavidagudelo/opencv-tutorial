from basic import histograms
from basic import load_images

img = load_images.read_image('basic/images/image1.jpeg')
histograms.show_histograms(img)