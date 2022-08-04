import cv2
import numpy as np


def make_pdf(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    return hist


def transform_depth(depth):
    depth -= depth.min()
    # Scale by 2 mean distances of near rays.
    depth /= 2 * depth[depth <= 1].mean()
    # Scale to [0, 255]
    pixels = 255 * np.clip(depth, 0, 1)
    # For debugging
    # cv2.imwrite(f"depth_{camera}.png", pixels)
    return pixels
