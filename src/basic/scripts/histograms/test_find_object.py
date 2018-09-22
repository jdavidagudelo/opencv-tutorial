import cv2
from basic import histograms
from basic import load_images
import os

path = '{0}/../../images/messi.jpg'.format(os.path.dirname(os.path.abspath(__file__)))
path1 = '{0}/../../images/grass.jpg'.format(os.path.dirname(os.path.abspath(__file__)))

roi = cv2.imread(path)
target = cv2.imread(path1)
res = histograms.get_merged_object(roi, target)
load_images.show_images([roi, res])

