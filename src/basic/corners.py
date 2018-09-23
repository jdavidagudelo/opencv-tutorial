import cv2
import numpy as np


def harris_corner_detection(img, red=255, green=0, blue=0, threshold=0.01, block_size=3, kernel_size=3, k=0.04):
    img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, block_size, kernel_size, k)

    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst > threshold * dst.max()] = [blue, green, red]
    return img


def get_good_features(img, max_corners, quality_level, min_distance):
    img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, max_corners, quality_level, min_distance)
    corners = np.int0(corners)
    for i in corners:
        x, y = i.ravel()
        img = cv2.circle(img, (x, y), 3, (255, 0, 0), 3)
    return img


def get_fast_features(img, threshold, no_max_suppression=True):
    img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fast = cv2.FastFeatureDetector_create(threshold, no_max_suppression)
    kp = fast.detect(gray, None)
    img2 = cv2.drawKeypoints(img, kp, color=(255, 0, 0), outImage=img)
    return img2


def get_orb(img, n_features=500, scale_factor=2):
    img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=n_features, scaleFactor=scale_factor)
    # find the keypoints with ORB
    kp = orb.detect(gray, None)
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    # draw only keypoints location,not size and orientation
    img2 = cv2.drawKeypoints(img, kp, color=(0, 255, 0), flags=0, outImage=img)
    return img2


def brute_force_matching(img, img_query, n_features=500, scale_factor=2, match_count=10):
    img = img.copy()
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img_query, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=n_features, scaleFactor=scale_factor)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:match_count],
                           flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS, outImg=img1)
    return img3

