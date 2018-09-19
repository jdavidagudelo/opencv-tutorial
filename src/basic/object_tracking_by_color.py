import cv2


def detect_object_by_hsv_color(img, low_color, high_color):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, low_color, high_color)
    res = cv2.bitwise_and(img, img, mask=mask)
    return img, mask, res
