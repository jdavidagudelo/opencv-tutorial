import cv2
import numpy as np


def find_contours(img, thresh, max_value):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, thresh, max_value, cv2.THRESH_BINARY)
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return image, contours, hierarchy


def draw_contours(img, contours, index, red, green, blue, thickness):
    return cv2.drawContours(img.copy(), contours, index, (blue, red, green), thickness=thickness)


def draw_contours_with_centroid(img, contours, index, red, green, blue, thickness, min_area=50):
    img = img.copy()
    if index > 0:
        contours = [contours[index]]
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            m = cv2.moments(cnt)
            cx = int(m['m10'] / m['m00'])
            cy = int(m['m01'] / m['m00'])
            img = cv2.circle(img, (cx, cy), 3, (blue, green, red), thickness=thickness)
    return cv2.drawContours(img, contours, index, (blue, red, green), thickness=thickness)


def draw_contours_with_approximation(img, contours, index, ratio, red, green, blue, thickness, min_area=50):
    img = img.copy()
    if index > 0:
        contours = [contours[index]]
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            epsilon = ratio * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            img = cv2.drawContours(img, [approx], 0, (blue, red, green), thickness=thickness)
    return img


def draw_contours_convex_hull(img, contours, index, red, green, blue, thickness, min_area=50):
    if index > 0:
        contours = [contours[index]]
    img = img.copy()
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            convex_hull = cv2.convexHull(cnt)
            img = cv2.drawContours(img, [convex_hull], 0, (blue, red, green), thickness=thickness)
    return img


def draw_contours_rectangles_straight(img, contours, index, red, green, blue, thickness, min_area=50):
    if index > 0:
        contours = [contours[index]]
    img = img.copy()
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (blue, green, red), thickness=thickness)
    return img


def draw_contours_rectangles_rotated(img, contours, index, red, green, blue, thickness, min_area=50):
    if index > 0:
        contours = [contours[index]]
    img = img.copy()
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            img = cv2.drawContours(img, [box], 0, (blue, green, red), thickness)
    return img


def draw_contours_circles(img, contours, index, red, green, blue, thickness, min_area=50):
    if index > 0:
        contours = [contours[index]]
    img = img.copy()
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            img = cv2.circle(img, center, radius, (blue, green, red), thickness)
    return img


def draw_contours_ellipses(img, contours, index, red, green, blue, thickness, min_area=50):
    if index > 0:
        contours = [contours[index]]
    img = img.copy()
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area and len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            img = cv2.ellipse(img, ellipse, (blue, red, green), thickness)
    return img


def draw_contours_lines(img, contours, index, red, green, blue, thickness, min_area=50):
    if index > 0:
        contours = [contours[index]]
    img = img.copy()
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            rows, cols = img.shape[:2]
            [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
            lefty = int((-x * vy / vx) + y)
            righty = int(((cols - x) * vy / vx) + y)
            img = cv2.line(img, (cols - 1, righty), (0, lefty), (blue, green, red), thickness)
    return img


def draw_convex_defects(img, threshold, max_value):
    img = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, threshold, max_value, 0)
    z, contours, hierarchy = cv2.findContours(thresh, 2, 1)
    cnt = contours[0]
    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)

    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        img = cv2.line(img, start, end, [0, 255, 0], 2)
        img = cv2.circle(img, far, 5, [0, 0, 255], -1)
    return img


def aspect_ratio(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return float(w) / h


def extent(contour):
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    rect_area = w * h
    return float(area) / rect_area


def solidity(contour):
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    return float(area) / hull_area


def equivalent_diameter(contour):
    area = cv2.contourArea(contour)
    return np.sqrt(4 * area / np.pi)


def orientation(contour):
    (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
    return angle


def extreme_points(contour):
    leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
    rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
    topmost = tuple(contour[contour[:, :, 1].argmin()][0])
    bottommost = tuple(contour[contour[:, :, 1].argmax()][0])
    return leftmost, rightmost, topmost, bottommost


def get_distance_contour(contour, x, y):
    return cv2.pointPolygonTest(contour, (x, y), True)


def match_shapes(contour1, contour2):
    return cv2.matchShapes(contour1, contour2, 1, 0.0)