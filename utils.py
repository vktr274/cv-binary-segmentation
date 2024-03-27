import numpy as np


def merge_image_superpixels(image, mask, color):
    """
    Merges the superpixel contour mask with the original 3-channel image.

    :param image: The original 3-channel image.
    :param mask: The superpixel contour mask.
    :param color: The color to use for the contours. E.g. [255, 0, 0] for blue (BGR format).
    :return: The merged image.
    """
    superpixel_contour_img = np.zeros(image.shape, dtype=np.uint8)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if mask[i, j] == 255:
                superpixel_contour_img[i, j] = color
            else:
                superpixel_contour_img[i, j] = image[i, j]

    return superpixel_contour_img
