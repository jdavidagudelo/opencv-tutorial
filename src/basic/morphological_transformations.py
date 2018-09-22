import cv2
import numpy as np


def erode(img, kernel=None, iterations=1):
    if kernel is None:
        kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(img, kernel, iterations=iterations)


def dilate(img, kernel=None, iterations=1):
    return cv2.dilate(img, kernel=kernel, iterations=iterations)


def image_open(img, kernel=None, iterations=1):
    if kernel is None:
        kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel=kernel, iterations=iterations)


def image_close(img, kernel=None, iterations=1):
    if kernel is None:
        kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel=kernel, iterations=iterations)


def morphological_gradient(img, kernel=None, iterations=1):
    if kernel is None:
        kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel=kernel, iterations=iterations)


def top_hat(img, kernel=None, iterations=1):
    if kernel is None:
        kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel=kernel, iterations=iterations)


def black_hat(img, kernel=None, iterations=1):
    if kernel is None:
        kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel=kernel, iterations=iterations)
