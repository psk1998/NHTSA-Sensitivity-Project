## Bounding Box Creation
import numpy as np
def coordinates(img):
    xmin = 1000
    ymin = 1000
    xmax = -1
    ymax = -1
    for y in np.arange(0,img.shape[0]):
        for x in np.arange(0,img.shape[1]):
            if img[y,x] != 0:
                ymin = min(ymin,y)
                xmin = min(xmin,x)
                ymax = max(ymax,y)
                xmax = max(xmax,x)
    return ymin, xmin, ymax, xmax

def draw(orig_img,xmin,ymin,xmax,ymax,color):
    import cv2
    cv2.rectangle(orig_img,(xmin,ymin),(xmax,ymax),color)
    cv2.imshow('BBox',orig_img)
    cv2.waitKey(0)


## References: https://gist.github.com/meyerjo/dd3533edc97c81258898f60d8978eddc

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


# Dice similarity function
def dice(pred, true, k = 1):
    intersection = np.sum(pred[true==k]) * 2.0
    dice = intersection / (np.sum(pred) + np.sum(true))
    return dice

# dice_score = dice(y_pred, y_true, k = 255) #255 in my case, can be 1 
# print ("Dice Similarity: {}".format(dice_score))

## https://gist.github.com/JDWarner/6730747
def dicecoef(im1, im2):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        
    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())

def IoU_Segmentation(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute IoU 
    intersection = np.logical_and(im1, im2)

    return intersection.sum() / (im1.sum() + im2.sum() - intersection.sum())