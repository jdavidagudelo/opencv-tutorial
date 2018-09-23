import cv2
import numpy as np


def get_magnitude_spectrum(img, multiplier=20):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = multiplier * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    return magnitude_spectrum


def get_inverse_dft(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    # create a mask first, center square is 1, remaining all zeros
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1
    # apply mask and inverse DFT
    f_shift = dft_shift * mask
    f_i_shift = np.fft.ifftshift(f_shift)
    img_back = cv2.idft(f_i_shift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    return img_back


def template_matching(img, template, method):
    img = img.copy()
    template = template.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    w, h = template.shape[::-1]
    # Apply template Matching
    res = cv2.matchTemplate(img_gray, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    return cv2.rectangle(img, top_left, bottom_right, 255, 2)


def template_matching_multiple(img, template, threshold, method=cv2.TM_CCOEFF_NORMED):
    img = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img_gray, template, method)
    threshold = threshold
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    return img


def get_hough_lines(img, min_value, max_value, aperture_size, rho=1, theta=np.pi/180, threshold=200):
    img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, min_value, max_value, apertureSize=aperture_size)
    lines = cv2.HoughLines(edges, rho, theta, threshold)
    for line in lines:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)
            img = cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return img


def get_hough_lines_probabilistic(img, min_value, max_value, aperture_size, rho=1, theta=np.pi/180, threshold=200):
    img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, min_value, max_value, apertureSize=aperture_size)
    min_line_length = 50
    max_lin_gap = 1
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, min_line_length, max_lin_gap)
    for x1, y1, x2, y2 in lines[0]:
        img = cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img


def get_hough_circles(img):
    cimg = cv2.medianBlur(img, 5)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=30, minRadius=0, maxRadius=0)

    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cimg = cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cimg = cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
    return cimg


def water_shed_images(img):
    img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]
    return img


