import cv2
import pydicom

import numpy as np

from pydicom.pixel_data_handlers.util import apply_modality_lut

VOXELS = None
WWIDTH = 300
WLOC = 1200
SHIFT = 2000

def apply_lut(dataset, voxels):
    if  any(x in dataset for x in ('ModalityLUTSequence', 'RescaleIntercept', 'RescaleSlope')):
        voxels = apply_modality_lut(voxels, dataset)

    return voxels


def get_pixels(voxels):
    voxels = voxels - np.min(voxels)
    voxels = voxels / np.max(voxels)
    pixels = 1. * voxels

    return pixels


def on_trackbar(val):
    global WLOC
    WLOC = val - SHIFT
    pixels = np.where(VOXELS < WLOC + WWIDTH, VOXELS, 255)
    pixels = np.where(pixels > WLOC - WWIDTH, pixels, 0)
    pixels = get_pixels(pixels)

    cv2.imshow('Window Example', pixels)

def on_trackwidth(val):
    global WWIDTH
    WWIDTH = val

    on_trackbar(WLOC + SHIFT)


def main() -> None:
    global VOXELS
    dataset = pydicom.dcmread('example.dcm')
    VOXELS = apply_lut(dataset, dataset.pixel_array)

    cv2.namedWindow('Window Example')
    cv2.createTrackbar('Window center', 'Window Example', 1200, 6000, on_trackbar)
    cv2.createTrackbar('Windows width', 'Window Example', 300, 1000, on_trackwidth)
    on_trackbar(1200)
    cv2.waitKey()


if __name__ == '__main__':
    main()
