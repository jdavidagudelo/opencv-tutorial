from basic import load_images
from basic import transforms
import cv2
import os

path = '{0}/../../images/messi.jpg'.format(os.path.dirname(os.path.abspath(__file__)))
path1 = '{0}/../../images/messi_face.jpg'.format(os.path.dirname(os.path.abspath(__file__)))
img = load_images.read_image(path)
template = load_images.read_image(path1)
methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
           cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]
images = []
for method in methods:
    images.append(transforms.template_matching(img, template, method))

load_images.show_images(images, 2, 3)
