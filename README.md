# Assignment 3

Viktor Modroczký\
Computer Vision @ FIIT STU

## Task 1 - GrabCut

The [`task1.ipynb`](task1.ipynb) notebook contains the code and visualizations for this task.

Selected image: dog on a chair\
Object of interest: the whole chair

### Steps

1. Draw a mask in Photopea.\
![Mask](data/mask.png)\
The red area marks possible foreground, the whie area marks sure foreground, and the black area marks sure background.

2. Load mask with OpenCV and convert it to a mask with values `cv2.GC_BGD` (black), `cv2.GC_FGD` (gray), and `cv2.GC_PR_FGD` (white) for sure background, sure foreground, and possible foreground, respectively.\
![Mask](images/my_mask.png)

3. Use the mask to initialize the GrabCut algorithm with 5 iterations. The result is shown below.\
![Result](images/grabcut_mask.png)

4. Mark sure foreground and possible foreground as sure foreground. The result is shown below in comparison with the ground truth.\
![Result](images/final_mask.png)

5. Apply the mask to the original image. The result is shown below.\
![Result](images/segmented.png)

6. Compute DICE coefficient. The resulting similarity is 96.73%.

## Task 2 - Semi-supervised Segmentation

The [`task2.ipynb`](task2.ipynb) notebook contains the implementation of the semi-supervised segmentation algorithm. The algorithm is also implemented in the [`utils.py`](utils.py) file as a function without visualization called `semisupervised_segmentation`. This version is then used for evaluation.

### Proposed algorithm

1. Load image and blur it with a median filter of size 7x7.
2. Apply superpixel segmentation with LSC algorithm with a region size of 16 and 20 iterations.
3. Get inverted superpixel contour mask and apply L2 distance transform. The distance transform is shown below for one of the images.\
![Superpixel mask and its distance transform](images/superpixels_and_dist_trans.png)
4. Compute the local maxima of the distance transform. The result is shown below.\
![Local maxima of distance transform](images/local_maxima.png)
5. Compute Delaunay triangulation of the local maxima using Scipy and create a graph with nodes as local maxima and edges as Delaunay triangles using NetworkX. The result is shown below.\
![Delaunay triangulation](images/delaunay.png)
6. Show image with superpixel contours overlayed with the original image. Allow the user to select points around an area of interest.
7. Find the closest points to the selected points in array of local maxima.
8. Compute the shortest path between these points using Dijkstra's algorithm. The result is shown below in comparison with the selected local maxima.\
![Shortest path](images/local_maxima_selected_found.png)
9. Create mask of superpixels (each dilated by a rectangular 11x11 structuring element in 1 iteration) that the shortest path traverses. The result is shown below.\
![Mask of superpixels](images/path_superpixels.png)
10. Dilate the mask in 4 iterations using an 7x7 elliptic structuring element. The result is shown below.\
![Dilated mask](images/dilated_mask.png)
11. Find contour of the mask and fill it. Compute XOR of the mask with the dilated mask which results in a mask with containing holes of the dilated mask. This way, we can mark the holes as sure foreground. The result is shown below.\
![Created mask](images/created_mask.png)
12. Apply grabcut with the created mask and 5 iterations. The resulting mask is shown below.\
![After grabcut](images/mask_after_grabcut.png)
13. Mark sure foreground and possible foreground as sure foreground. The result is shown below in comparison with the ground truth for one of the images. The DICE score between the mask after grabcut and ground truth mask is 91.74%.\
![Final mask](images/final_mask_2.png)
14. Apply the mask to the original image. The result is shown below for one of the images.\
![Segmented image](images/segmented_2.png)

### Algorithm Evaluation

The [`segmentation_test.ipynb`](segmentation_test.ipynb) notebook contains the evaluation of the proposed algorithm. The algorithm is evaluated on 3 more images.

#### Image 1

The DICE score between the mask after grabcut and ground truth mask is 88.84%.

![Segmented image](images/eval_1.png)

#### Image 2

The DICE score between the mask after grabcut and ground truth mask is 93.58%.

![Segmented image](images/eval_2.png)

#### Image 3

The DICE score between the mask after grabcut and ground truth mask is 96.82%.

![Segmented image](images/eval_3.png)

## Task 3 - Moving Object Segmentation
