import cv2
import numpy


def average_filter(img, kernel_cols, kernel_rows):
    kernel = numpy.ones((kernel_cols, kernel_rows), numpy.float32) / (kernel_rows * kernel_cols)
    return cv2.filter2D(img, -1, kernel)


def average_blur(img, kernel_rows, kernel_cols):
    return cv2.blur(img, (kernel_rows, kernel_cols))


def gaussian_blur(img, kernel_rows, kernel_cols, sigma_x=0, sigma_y=0):
    return cv2.GaussianBlur(img, (kernel_rows, kernel_cols), sigmaX=sigma_x, sigmaY=sigma_y)


def median_blur(img, kernel_size):
    return cv2.medianBlur(img, kernel_size)


def bilateral_filtering(img, diameter, sigma_color, sigma_space):
    return cv2.bilateralFilter(img, diameter, sigma_color, sigma_space)


def add_noise_to_image(img, prob=0.5):
    rows, cols, channels = img.shape
    rnd = numpy.random.rand(rows, cols)
    noisy = img.copy()
    noisy[rnd < prob] = numpy.random.randint(255)
    return noisy
