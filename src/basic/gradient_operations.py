import cv2


def laplacian(img):
    return cv2.Laplacian(img, cv2.CV_64F)


def sobel(img, dx, dy, kernel_size):
    return cv2.Sobel(img, cv2.CV_64F, dx, dy, ksize=kernel_size)


def canny(img, min_value, max_value):
    return cv2.Canny(img, min_value, max_value)
