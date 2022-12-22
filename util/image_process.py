import cv2
import numpy as np

from util import utils


def apply_binary_mask(image, mask):
    """Both input have to have 3 channels."""
    output_image = np.copy(image)
    height, width = image.shape[0], image.shape[1]
    _mask = cv2.resize(mask, (width, height))
    normalized_mask = _mask / 255
    output_image = np.multiply(output_image, normalized_mask)
    return output_image.astype(np.uint8)


def remove_green_pixels(img):
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([60, 255, 255])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.bitwise_not(mask, mask)
    res = cv2.bitwise_and(img, img, mask=mask)
    return res


def remove_black_pixels(img):
    color_mask = np.any(img != [0, 0, 0], axis=-1)
    return img[color_mask]


def bgr_to_hsv_color(img):
    return np.array([bgr_to_hsv(b, g, r) for b, g, r in img])


def bgr_to_hsv(b, g, r):
    color = np.uint8([[[b, g, r]]])
    hsv_colors = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    return int(hsv_colors[0][0][0]), int(hsv_colors[0][0][1]), int(hsv_colors[0][0][2])


def hsv_to_bgr(h, s, v):
    color = np.uint8([[[h, s, v]]])
    bgr_colors = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)
    return int(bgr_colors[0][0][0]), int(bgr_colors[0][0][1]), int(bgr_colors[0][0][2])


def bgr_to_spherical(b, g, r):
    rho = np.sqrt(np.square(b) + np.square(g) + np.square(r))
    theta = np.arctan2(np.sqrt(np.square(b) + np.square(g)), r)
    phi = np.arctan2(g, b)
    return float(rho), float(theta), float(phi)


def spherical_to_bgr(rho, theta, phi):
    b = rho * np.sin(theta) * np.cos(phi)
    g = rho * np.sin(theta) * np.sin(phi)
    r = rho * np.cos(theta)
    return int(round(b, 0)), int(round(g, 0)), int(round(r, 0))


def bgr_to_hex(bgr):
    return '#%02x%02x%02x' % (int(bgr[2]), int(bgr[1]), int(bgr[0]))


def hsv_to_hex(hsv):
    bgr = hsv_to_bgr(hsv[0], hsv[1], hsv[2])
    return '#%02x%02x%02x' % (int(bgr[2]), int(bgr[1]), int(bgr[0]))


def get_hue_histogram(sample):
    img = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
    channels = cv2.split(img)
    hist = cv2.calcHist([channels[0]], [0], None, [36], [0, 179])
    hist[0, :] = 0
    return hist.flatten()


def get_bgr_histogram(sample, bins=(32, 32, 32)):
    channels = cv2.split(sample)
    histogram = []
    for i, channel in enumerate(channels):
        hist = cv2.calcHist([channel], [0], None, [bins[i]], [0, 256])
        hist[0, :] = 0
        histogram.extend(hist)
    histogram = np.array(histogram)
    return histogram.flatten()


def get_hsv_histogram(sample, bins=(36, 32, 32)):
    img = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
    channels = cv2.split(img)
    histogram = []
    hue_hist = cv2.calcHist([channels[0]], [0], None, [bins[0]], [0, 180])
    hue_hist[0, :] = 0
    histogram.extend(hue_hist)
    for i in range(1, 3):
        hist = cv2.calcHist([channels[i]], [0], None, [bins[i]], [0, 256])
        hist[0, :] = 0
        histogram.extend(hist)
    histogram = np.array(histogram)
    return histogram.flatten()
