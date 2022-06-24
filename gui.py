import cv2
import pydicom
import numpy as np

from pydicom.pixel_data_handlers.util import apply_modality_lut


def apply_window(dataset, voxels, window: tuple):
    if any(x in dataset for x in ('ModalityLUTSequence', 'RescaleIntercept', 'RescaleSlope')):
        voxels = apply_modality_lut(voxels, dataset)

    # Вставить проверку на инверсию пикселей

    voxels = np.where(voxels < window[0], window[0], voxels)
    voxels = np.where(voxels > window[1], window[1], voxels)

    voxels = voxels - np.min(voxels)
    voxels = voxels / np.max(voxels)
    pixels = (255. * voxels).astype(np.uint8)

    return pixels


def windows():
    dataset = pydicom.dcmread('example.dcm')
    voxels = dataset.pixel_array

    windows = dict(
        LUNG=(-1250, 250),
        MEDIASTINUM=(-120, 280),
        BONES=(-650, 1350),
        ABDOMEN=(-110, 210)
    )

    for window in windows.keys():
        pixels = apply_window(dataset, voxels, windows[window])
        cv2.imwrite(f'{window}.jpg', pixels)


if __name__ == '__main__':
    # main()
    windows()
