import cv2
import pydicom

import numpy as np

from pydicom.pixel_data_handlers.util import apply_voi_lut, apply_modality_lut

VOXELS = None


def apply_lut(dataset, voxels):
    if 'ModalityLUTSequence' in dataset:
        voxels = apply_modality_lut(voxels, dataset)
        voxels = apply_voi_lut(voxels, dataset)
    else:
        voxels = apply_voi_lut(voxels, dataset)

    return voxels


def get_pixels(voxels):
    voxels = voxels - np.min(voxels)
    voxels = voxels / np.max(voxels)
    pixels = 255 * voxels

    return pixels


def on_trackbar(val):
    pixels = np.where(VOXELS < val + 1000, VOXELS, 0)
    pixels = np.where(pixels > val - 1000, pixels, 0)
    pixels = get_pixels(pixels)

    cv2.imshow('Window Example', pixels)


def main() -> None:
    global VOXELS
    dataset = pydicom.dcmread('example.dcm')
    VOXELS = apply_lut(dataset, dataset.pixel_array)

    cv2.namedWindow('Window Example')
    trackbar_name = 'Window center'
    cv2.createTrackbar(trackbar_name, 'Window Example', 0, 4000, on_trackbar)
    # Show some stuff
    on_trackbar(0)
    # Wait until user press some key
    cv2.waitKey()


if __name__ == '__main__':
    main()
