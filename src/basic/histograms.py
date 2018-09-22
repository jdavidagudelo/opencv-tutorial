import cv2
import numpy as np
from matplotlib import pyplot as plt


def show_histograms(img):
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histogram = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histogram, color=col)
        plt.xlim([0, 256])
    plt.show()


def equalize_histogram(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    equ = cv2.equalizeHist(img)
    return equ


def create_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)


def get_hsv_histogram(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])


def get_merged_object(img, internal_img):
    hsv = cv2.cvtColor(internal_img, cv2.COLOR_BGR2HSV)
    hsvt = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # calculating object histogram
    roihist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

    # normalize histogram and apply backprojection
    cv2.normalize(roihist, roihist, 0, 255, cv2.NORM_MINMAX)
    dst = cv2.calcBackProject([hsvt], [0, 1], roihist, [0, 180, 0, 256], 1)

    # Now convolve with circular disc
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cv2.filter2D(dst, -1, disc, dst)

    # threshold and binary AND
    ret, thresh = cv2.threshold(dst, 50, 255, 0)
    thresh = cv2.merge((thresh, thresh, thresh))
    res = cv2.bitwise_and(img, thresh)
    return res
