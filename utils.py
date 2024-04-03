import copy
import numpy as np
import cv2
from scipy.spatial import Delaunay
import networkx as nx
from networkx.algorithms.shortest_paths.weighted import dijkstra_path


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


def mouse_callback(event, x, y, _, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        x_math = x
        y_math = param[0].shape[0] - y
        print(f"Left button clicked at ({x_math}, {y_math})")
        param[1].append(np.array([x_math, y_math]))


def semisupervised_segmentation(image):
    """
    Implements the algorithm proposed in the task2.ipynb notebook.
    """
    median_blurred = cv2.medianBlur(image, 7)
    lab_img = cv2.cvtColor(median_blurred, cv2.COLOR_BGR2Lab)

    superpixels_lsc = cv2.ximgproc.createSuperpixelLSC(lab_img, region_size=16)
    superpixels_lsc.iterate(20)
    superpixels_lsc_mask = superpixels_lsc.getLabelContourMask()

    superpixel_contour_img = merge_image_superpixels(
        image, superpixels_lsc_mask, [0, 255, 0]
    )

    superpixel_labels = superpixels_lsc.getLabels()
    masks = []

    for i in np.unique(superpixel_labels):
        mask = np.where(superpixel_labels == i, 1, 0).astype(np.uint8)
        masks.append(mask)

    dist_trans = cv2.distanceTransform(~superpixels_lsc_mask, cv2.DIST_L2, 5)  # type: ignore

    local_maxima_coord = []

    for mask in masks:
        masked = mask * dist_trans
        local_maximum = np.argwhere(masked == np.max(masked))[0]
        dist_trans[local_maximum[0], local_maximum[1]] = 1

        local_maxima_coord.append(
            np.array([local_maximum[1], (dist_trans.shape[0] - 1) - local_maximum[0]])
        )

    local_maxima_coord = np.array(local_maxima_coord)

    trangulation = Delaunay(local_maxima_coord)
    G = nx.Graph()
    for path in trangulation.simplices:
        nx.add_path(G, path)

    clicks = []
    cv2.imshow("Image", superpixel_contour_img)
    cv2.setMouseCallback("Image", mouse_callback, (dist_trans, clicks))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    distances = []
    nearest_to_clicks = []

    for i, point_click in enumerate(clicks):
        for j, point_maximum in enumerate(local_maxima_coord):
            distances.append(np.array([i, j, cv2.norm(point_maximum, point_click)]))

    distances = np.array(distances)

    for i in range(len(clicks)):
        interest_points = distances[distances[:, 0] == i]
        minimum_dist = interest_points[
            interest_points[:, 2] == np.min(interest_points[:, 2]), 1
        ]
        nearest_to_clicks.append(int(minimum_dist[0]))

    indices_of_nearest = []

    for coord_click in local_maxima_coord[nearest_to_clicks]:
        for i, coord in enumerate(local_maxima_coord):
            if (coord == coord_click).all():
                indices_of_nearest.append(i)
                break

    indices_of_nearest = np.array(indices_of_nearest)
    result = copy.deepcopy(local_maxima_coord[nearest_to_clicks])

    for i in range(len(indices_of_nearest) - 1):
        path = dijkstra_path(G, indices_of_nearest[i], indices_of_nearest[i + 1])
        result = np.append(result, values=local_maxima_coord[path][1:-1], axis=0)  # type: ignore

    path = dijkstra_path(G, indices_of_nearest[0], indices_of_nearest[-1])
    result = np.append(result, values=local_maxima_coord[path][1:-1], axis=0)  # type: ignore

    selected_superpixels = np.zeros(dist_trans.shape, np.uint8)
    struct_elem = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))

    for mask in masks:
        for local_max in result:
            if mask[(dist_trans.shape[0] - 1) - local_max[1], local_max[0]] == 1:
                selected_superpixels |= cv2.dilate(mask, struct_elem, iterations=1)  # type: ignore

    struct_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated_selection = cv2.dilate(selected_superpixels, struct_elem, iterations=4)

    contours, _ = cv2.findContours(
        dilated_selection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    mask = np.zeros(dilated_selection.shape, dtype=np.uint8)
    mask = cv2.fillPoly(mask, contours, 1)  # type: ignore

    dilated_selection[dilated_selection == 1] = cv2.GC_PR_FGD

    certain_fg = np.logical_xor(mask, dilated_selection)
    certain_fg[certain_fg == 1] = cv2.GC_FGD

    created_mask = dilated_selection + certain_fg
    grabcut_mask = dilated_selection + certain_fg

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    grabcut_mask, bgd_model, fgd_model = cv2.grabCut(
        image, grabcut_mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK  # type: ignore
    )
    grabcut_mask = np.where(
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD), 1, 0
    )

    segmented = image * cv2.merge([grabcut_mask, grabcut_mask, grabcut_mask])

    return segmented, grabcut_mask, created_mask
