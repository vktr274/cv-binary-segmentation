# Assignment 3

Viktor Modroczký\
Computer Vision @ FIIT STU

## Task 1 - GrabCut

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
