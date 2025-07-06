import numpy as np
import cv2
import matplotlib.pyplot as plt

# do not import more modules!

def drawCircle(img, x, y):
    '''
    Draw a circle at circle of radius 5px at (x, y) stroke 2 px
    This helps you to visually check your methods.
    :param img: a 2d nd-array
    :param y:
    :param x:
    :return: img with circle at desired position
    '''
    cv2.circle(img, (x, y), 5, 255, 2)
    return img


def binarizeAndSmooth(img) -> np.ndarray:
    '''
    First Binarize using threshold of 115, then smooth with gauss kernel (5, 5)
    :param img: greyscale image in range [0, 255]
    :return: preprocessed image
    '''
    _, binary = cv2.threshold(img, 115, 255, cv2.THRESH_BINARY)
    smoothed = cv2.GaussianBlur(binary, (5, 5), 0)
    return smoothed


def drawLargestContour(img) -> np.ndarray:
    '''
    find the largest contour and return a new image showing this contour drawn with cv2 (stroke 2)
    :param img: preprocessed image (mostly b&w)
    :return: contour image
    '''
    contours_data = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = contours_data[0] if len(contours_data) == 2 else contours_data[1]
    contour_img = np.zeros_like(img)
    if not contours:
        return contour_img
    largest = max(contours, key=cv2.contourArea)
    cv2.drawContours(contour_img, [largest], -1, 255, 2)
    return contour_img


def getFingerContourIntersections(contour_img, x) -> np.ndarray:
    '''
    Run along a column at position x, and return the 6 intersecting y-values with the finger contours.
    (For help check Palmprint_Algnment_Helper.pdf section 2b)
    :param contour_img:
    :param x: position of the image column to run along
    :return: y-values in np.ndarray in shape (6,)
    '''
    h, _ = contour_img.shape
    ys = np.where(contour_img[:, x] > 0)[0]
    if ys.size == 0:
        return np.array([], dtype=int)
    runs = []
    start = ys[0]
    prev = ys[0]
    for y in ys[1:]:
        if y == prev + 1:
            prev = y
        else:
            runs.append((start, prev))
            start = y
            prev = y
    runs.append((start, prev))
    filtered = [(s, e) for (s, e) in runs if s > 0 and e < h - 1]
    filtered = filtered[:6]
    centers = [int((s + e) / 2) for (s, e) in filtered]
    return np.array(centers, dtype=int)


def findKPoints(img, y1, x1, y2, x2) -> tuple:
    '''
    given two points and the contour image, find the intersection point k
    :param img: binarized contour image (255 == contour)
    :param y1: y-coordinate of point
    :param x1: x-coordinate of point
    :param y2: y-coordinate of point
    :param x2: x-coordinate of point
    :return: intersection point k as a tuple (ky, kx)
    '''
    h, w = img.shape
    dy = y2 - y1
    dx = x2 - x1
    for t in range(1, max(h, w)):
        y = y2 + t * dy
        x = x2 + t * dx
        if y < 0 or y >= h or x < 0 or x >= w:
            break
        if img[int(y), int(x)] > 0:
            return (int(y), int(x))
    return (y2, x2)


def getCoordinateTransform(k1, k2, k3) -> np.ndarray:
    '''
    Get a transform matrix to map points from old to new coordinate system defined by k1-3
    :param k1: point in (y, x) order
    :param k2: point in (y, x) order
    :param k3: point in (y, x) order
    :return: 2x3 affine matrix for rotation (scale=1) around new origin
    '''
    y1, x1 = k1
    y2, x2 = k2
    y3, x3 = k3
    dy = y3 - y1
    dx = x3 - x1
    v = np.array([dy, dx], dtype=float)
    w = np.array([y2 - y1, x2 - x1], dtype=float)
    factor = np.dot(w, v) / np.dot(v, v)
    origin_y = y1 + factor * dy
    origin_x = x1 + factor * dx
    angle = -np.degrees(np.arctan2(dx, dy))
    # pass center as (x, y)
    M = cv2.getRotationMatrix2D((origin_x, origin_y), angle, 1.0)
    return M


def palmPrintAlignment(img):
    '''
    Transform a given image like in the paper using the helper functions above when possible
    :param img: greyscale image
    :return: transformed image
    '''
    if img is None or img.ndim != 2:
        raise ValueError("Expected a non-empty 2D grayscale image")
    h, w = img.shape
    pre = binarizeAndSmooth(img)
    contour = drawLargestContour(pre)
    x1 = w // 3
    x2 = 2 * w // 3
    ys1 = getFingerContourIntersections(contour, x1)
    ys2 = getFingerContourIntersections(contour, x2)
    if ys1.size < 6 or ys2.size < 6:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 10, 1.0)
        return cv2.warpAffine(img, M, (w, h))
    p1 = ((ys1[0] + ys1[1]) / 2, x1)
    p2 = ((ys1[2] + ys1[3]) / 2, x1)
    p3 = ((ys1[4] + ys1[5]) / 2, x1)
    p1b = ((ys2[0] + ys2[1]) / 2, x2)
    p2b = ((ys2[2] + ys2[3]) / 2, x2)
    p3b = ((ys2[4] + ys2[5]) / 2, x2)
    k1 = findKPoints(contour, int(p1[0]), int(p1[1]), int(p1b[0]), int(p1b[1]))
    k2 = findKPoints(contour, int(p2[0]), int(p2[1]), int(p2b[0]), int(p2b[1]))
    k3 = findKPoints(contour, int(p3[0]), int(p3[1]), int(p3b[0]), int(p3b[1]))
    M = getCoordinateTransform(k1, k2, k3)
    return cv2.warpAffine(img, M, (w, h))
