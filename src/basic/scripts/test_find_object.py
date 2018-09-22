import cv2
from basic import histograms
from basic import load_images

roi = cv2.imread('basic/images/messi.jpg')
target = cv2.imread('basic/images/grass.jpg')
res = histograms.get_merged_object(roi, target)
load_images.show_images([roi, res])

