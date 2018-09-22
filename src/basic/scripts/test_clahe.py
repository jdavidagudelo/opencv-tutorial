from basic import load_images
from basic import histograms

img = load_images.read_image('basic/images/image1.jpeg')
new_img = histograms.create_clahe(img)
load_images.show_images([img, new_img])
