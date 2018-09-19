import cv2
import numpy


def get_pixel(img, x, y):
    return img[x, y]


def get_color(img, x, y, color):
    return img.item(x, y, color)


def set_color(img, x, y, color, value):
    img.itemset(x, y, color, value)


def get_shape(img):
    return img.shape


def number_of_pixels(img: numpy.array):
    return img.size


def get_dtype(img: numpy.array):
    return img.dtype


def copy_region(img, another_img, x0, y0, x1, y1):
    img[x0:x1, y0:y1] = another_img


def split(img):
    red, green, blue = cv2.split(img)
    return red, green, blue


def merge_components(red, green, blue):
    img = cv2.merge((red, green, blue))
    return img


def make_border(img, top, bottom, left, right, border_type=cv2.BORDER_WRAP):
    return cv2.copyMakeBorder(img, top, bottom, left, right, border_type)


def add_images(img1, img2):
    return cv2.add(img1, img2)


def add_images_weighted(img1, img2, alpha, beta, gamma):
    return cv2.addWeighted(img1, alpha, img2, beta, gamma)


def resize_image(img, rows, columns):
    return cv2.resize(img, (rows, columns))


def make_same_size(img1, img2):
    return resize_image(img1, img2.shape[1], img2.shape[0])


def add_image_bitwise(img1, img2, x0, y0, threshold=10, max_value=255):
    rows, cols, channels = img2.shape
    roi = img1[x0:rows, y0:cols]
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, threshold, max_value, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    dst = cv2.add(img1_bg, img2_fg)
    img1[x0:rows, y0:cols] = dst
    return img1
