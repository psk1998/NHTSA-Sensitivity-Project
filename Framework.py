### Trial Code for loading images using CV2 ###

# Import Required Packages
import cv2
import numpy as np
import copy
import math
import xml.etree.ElementTree as ET
from skimage.color import rgb2gray
from PIL import Image, ImageEnhance

from Libraries import Detection_Algorithm
from Libraries.AugmentationLibraries.ARALmaster import Automold as am
from Libraries.AugmentationLibraries.ARALmaster import Helpers as hp
from Libraries.Semantic import SemanticSegmentation
from Libraries.Sensitivity import Norm
from Libraries import bbox
from Libraries.image_similarity_measures_master.image_similarity_measures.quality_metrics import rmse, fsim
from skimage.metrics import structural_similarity as ssim

print('Packages Imported')

## Importing Image
# Add Image to the framework
print('Please Choose which dataset you would like to use:')
print('1: KITTI')
print('2: Cityscapes')
print('3: Pascal VOC')
img_src = input('Enter the number:')

print('Please Choose which algorithm you would like to test:')
print('1: Deeplab')
print('2: FCN')
print('3: MobileNet')
print('4: Lrass MobileNet')
print('5: PSPnet101_cityscapes')
print('6: PSPnet101_voc')
print('7: PSPnet101_ADE')
Algo = input('Enter the number:')

if img_src == '1':
    file = '000965'
    img_path = 'D:/SP 21/Project/Datasets/KITTI/Images/training/image_2/'+file+'.png'
    file = '000155_10'
    # file = '000041_10'
    # file = '000006_10'
    # file = '000058_10'
    img_path = 'D:/SP 21/Project/Datasets/KITTI/Semantic/training/image_2/'+file+'.png'
    gt_path = 'D:/SP 21/Project/Datasets/KITTI/Semantic/training/semantic_rgb/'+file+'.png'
#    gt_bbox_path = ''
elif img_src == '2':
    city = 'munich'
    file = '_000296_000019_'
    img_path = 'D:/SP 21/Project/Datasets/Cityscapes/leftImg8bit/test/'+city+'/'+city+file+'leftImg8bit.png'
elif img_src == '3':
    img_path = 'D:/SP 21/Project/Datasets/VOC2007/JPEGImages/003143.jpg'
    gt_path = 'D:/SP 21/Project/Datasets/VOC2007/SegmentationClass/003143.png'
    gt_bbox_path = 'D:/SP 21/Project/Datasets/VOC2007/Annotations/003143.xml'

orig_img = cv2.imread(img_path)
scalingFactor = 256/orig_img.shape[0]
orig_img_ynew = int(scalingFactor*orig_img.shape[1])
orig_img = cv2.resize(orig_img,(orig_img_ynew,256))

img_original = copy.deepcopy(orig_img)
img_perturbated = copy.deepcopy(orig_img)

# Original GT is of some other scale, we need to rescale the image such that the height is 256 pixels (the algorithms used are on 256 height)
img_gt = cv2.imread(gt_path)
gtxsize = img_gt.shape[0]
gtysize = img_gt.shape[1]
img_gtynew = int((256/gtxsize)*gtysize)
img_gt = cv2.resize(img_gt,(img_gtynew,256)) # Height x Width

# GT Sementation
# cv2.imshow('Segmentation GT',img_gt)
# cv2.waitKey(0)
img_gt = cv2.cvtColor(img_gt,cv2.COLOR_BGR2RGB)

## Image Augmentation/Pertubation
print('Please Choose which type of pertubation you want to perform:')
print('1: Brightness')
print('2: Darkness')
print('3: Fog')
print('4: Snow')
print('5: Rain')
print('6: Motion Blur')
print('7: Contrast')
print('8: Sharpness')
print('9: Color')
pertub = input('Pertubation:')

if pertub == '1':
    # Image Pertubtion Brighten
    factor_perturbation = float(input('Enter the factor by which image must be perturbed:'))
    img_perturbated = am.brighten(img_perturbated,brightness_coeff=factor_perturbation)

elif pertub == '2':
    # Image Pertubtion Darken
    factor_perturbation = float(input('Enter the factor by which image must be perturbed:'))
    img_perturbated = am.darken(img_perturbated,darkness_coeff=factor_perturbation)

elif pertub == '3':
    # Fog Addition
    factor_perturbation = float(input('Enter the factor by which image must be perturbed:'))
    img_perturbated = am.add_fog(img_perturbated,fog_coeff=factor_perturbation)

elif pertub == '4':
    # Snow Addition
    factor_perturbation = float(input('Enter the factor by which image must be perturbed:'))
    img_perturbated = am.add_snow(img_perturbated,snow_coeff=factor_perturbation)
    
elif pertub == '5':
    # Rain Addition
    img_perturbated = am.add_rain(img_perturbated,-1,5,2,(200,200,200),'None')
    img_perturbated = am.add_fog(img_perturbated,fog_coeff=0.2)
    factor_perturbation = 1

elif pertub == '6':
    # Motionblur Addition
    factor_perturbation = 2
    img_perturbated = am.apply_motion_blur(img_perturbated,2)

elif pertub == '7':
    # Contrast
    factor_perturbation = 1 + float(input('Enter the factor by which image must be perturbed:'))
    # img_perturbated = am.apply_motion_blur(img_perturbated,2)
    im = Image.open(img_path)
    enhancer = ImageEnhance.Contrast(im)
    im_output = enhancer.enhance(factor_perturbation)
    im_output.save('pertub.png')
    img_perturbated = cv2.imread('pertub.png')
    img_perturbated = cv2.resize(img_perturbated,(orig_img_ynew,256))

elif pertub == '8':
    # Sharpness
    factor_perturbation = 1 + float(input('Enter the factor by which image must be perturbed:'))
    im = Image.open(img_path)
    enhancer = ImageEnhance.Sharpness(im)
    im_output = enhancer.enhance(factor_perturbation)
    im_output.save('pertub.png')
    img_perturbated = cv2.imread('pertub.png')
    img_perturbated = cv2.resize(img_perturbated,(orig_img_ynew,256))

elif pertub == '9':
    # Color
    factor_perturbation = 1 - float(input('Enter the factor by which image must be perturbed:'))
    im = Image.open(img_path)
    enhancer = ImageEnhance.Color(im)
    im_output = enhancer.enhance(factor_perturbation)
    im_output.save('pertub.png')
    img_perturbated = cv2.imread('pertub.png')
    img_perturbated = cv2.resize(img_perturbated,(orig_img_ynew,256))

elif pertub == '10':
    # Brightness
    factor_perturbation = 1 + float(input('Enter the factor by which image must be perturbed:'))
    im = Image.open(img_path)
    enhancer = ImageEnhance.Brightness(im)
    im_output = enhancer.enhance(factor_perturbation)
    im_output.save('pertub.png')
    img_perturbated = cv2.imread('pertub.png')
    img_perturbated = cv2.resize(img_perturbated,(orig_img_ynew,256))

# Display Image
cv2.imshow('Image0',img_original)
cv2.waitKey(1)
cv2.imshow('Image1',img_perturbated)
cv2.waitKey(1)

# Save Images
cv2.imwrite('original.png',img_original)
cv2.imwrite('pertub.png',img_perturbated)

# IQA Assesment
# FSIM
fsim_INP = 1 - fsim(img_original,img_perturbated)
print('Input is:')
print(fsim_INP)

# SSIM
ssim_INP = 1 - ssim(img_original,img_perturbated,multichannel=True)
print('Input is:')
print(ssim_INP)

## Running Segmentation Algorithms
if Algo == '1':
    img_original = SemanticSegmentation.deeplab(img_original)
    img_perturbated = SemanticSegmentation.deeplab(img_perturbated)

elif Algo == '2':
    img_original = SemanticSegmentation.fcn(img_original)
    img_perturbated = SemanticSegmentation.deeplab(img_perturbated)

elif Algo == '3':
    img_original = SemanticSegmentation.mobilenet(img_original)
    img_perturbated = SemanticSegmentation.mobilenet(img_perturbated)

elif Algo == '4':
    img_original = SemanticSegmentation.lraspp_mobilenet(img_original)
    img_perturbated = SemanticSegmentation.lraspp_mobilenet(img_perturbated)

elif Algo == '5':
    img_original = SemanticSegmentation.pspnet101_cityscapes(img_original)
    img_perturbated = SemanticSegmentation.pspnet101_cityscapes(img_perturbated)

elif Algo == '6':
    img_original = SemanticSegmentation.pspnet101_voc(img_original)
    img_perturbated = SemanticSegmentation.pspnet101_voc(img_perturbated)

elif Algo == '7':
    img_original = SemanticSegmentation.pspnet101_ADE(img_original)
    img_perturbated = SemanticSegmentation.pspnet101_ADE(img_perturbated)

## This is valid only for the above 4 images
print(img_original.shape)
print(img_original[175][250])
print(img_original[150][250])
print(img_original[125][250])

print(img_perturbated.shape)
print(img_perturbated[175][250])
print(img_perturbated[150][250])
print(img_perturbated[125][250])

print(img_gt.shape)
print(img_gt[175][250])
print(img_gt[150][250])
print(img_gt[125][250])

### Extracting only a single class detetction ###
## Note that this is changing the image completely so is suggested to run in loop as a function
ref_gt = img_gt[150][250]
ref_imo = img_original[150][250]
ref_imp = img_perturbated[150][250]

ref_gt = [0,0,142]
ref_imo = [51,181,222]
ref_imp = [51,181,222]

for i in np.arange(img_gt.shape[0]):
    for j in np.arange(img_gt.shape[1]):
        if img_gt[i][j][0] != ref_gt[0] and img_gt[i][j][1] != ref_gt[1] and img_gt[i][j][2] != ref_gt[2]:
            img_gt[i][j][:] = [0,0,0]
        if img_original[i][j][0] != ref_imo[0] and img_original[i][j][1] != ref_imo[1] and img_original[i][j][2] != ref_imo[2]:
            img_original[i][j][:] = [0,0,0]
        if img_perturbated[i][j][0] != ref_imp[0] and img_perturbated[i][j][1] != ref_imp[1] and img_perturbated[i][j][2] != ref_imp[2]:
            img_perturbated[i][j][:] = [0,0,0]

## Norm Calculation
L1_original = Norm.L1(img_original,img_gt)
L1_perubated = Norm.L1(img_perturbated,img_gt)
L1_relative = Norm.L1(img_perturbated,img_original)

print('\nL1 norm of error of Original Image with respect to the ground truth is:')
print(L1_original)
print('\nL1 norm of error of Perturbated Image with respect to the ground truth is:')
print(L1_perubated)
print('\nL1 norm of error of Perturbated Image with respect to the Original Image is:')
print(L1_relative)

sensitivity = ((L1_perubated-L1_original)/L1_original)/(factor_perturbation)
print('\nSenitivity of L1 Norm is:')
print(sensitivity)

##### This can be deleted #####
## Convert from RGB to Grayscale
GS_img_original = rgb2gray(img_original)
GS_img_perturbated = rgb2gray(img_perturbated)

# Ground Truth Semantic Segmentation Mask
if img_src==1:
    dummy_gt = np.zeros(img_gt.shape)
    for i in img_gt.shape[0]:
        for j in img_gt.shape[1]:
            if img_gt[i][j] == [0,0,142]:
                dummy_gt[i][j] = [0,0,142]
            else:
                img_gt[i][j] == [0,0,0]
    img_gt = dummy_gt
GS_img_gt = rgb2gray(img_gt)

cv2.imshow('Image Output',GS_img_original)
cv2.waitKey(1)
cv2.imshow('Pertubed Output',GS_img_perturbated)
cv2.waitKey(1)
cv2.imshow('GT',GS_img_gt)
cv2.waitKey(1)

########## Metrics
## Boundingbox creation
## This works only for single object detection
## Drawing Bounding Box
ymin, xmin, ymax, xmax = bbox.coordinates(GS_img_original)
bbox.draw(orig_img,xmin,ymin,xmax,ymax,(255,0,0))
Box1 = [xmin,ymin,xmax,ymax]

ymin, xmin, ymax, xmax = bbox.coordinates(GS_img_perturbated)
bbox.draw(orig_img,xmin,ymin,xmax,ymax,(0,255,0))
Box2 = [xmin,ymin,xmax,ymax]

# Bbox from bbox GT
if img_src == '3':
    img_gt_bbox = ET.parse(gt_bbox_path)
    myRoot = img_gt_bbox.getroot()
    xmin = int(scalingFactor*float(myRoot[6][4][0].text))
    ymin = int(scalingFactor*float(myRoot[6][4][1].text))
    xmax = int(scalingFactor*float(myRoot[6][4][2].text))
    ymax = int(scalingFactor*float(myRoot[6][4][3].text))
    # ymin, xmin, ymax, xmax = bbox.coordinates(GS_img_gt)
    Box3 = [xmin,ymin,xmax,ymax]
    bbox.draw(orig_img,xmin,ymin,xmax,ymax,(0,0,255))

    ## IoU Computation
    IoU_Original_Img = bbox.bb_intersection_over_union(Box1,Box3)
    IoU_Perturbated_Img = bbox.bb_intersection_over_union(Box2,Box3)

    print('\nIoU of the Original Image is:')
    print(IoU_Original_Img)
    print('\nIoU of the Perturbated Image is:')
    print(IoU_Perturbated_Img)
    print('\nSensitivity of IoU is:')
    sensitivity = ((IoU_Perturbated_Img-IoU_Original_Img)/IoU_Original_Img)/(factor_perturbation)
    sensitivity = ((IoU_Perturbated_Img-IoU_Original_Img)/IoU_Original_Img)/(INP)
    print(sensitivity)
##### This can be deleted #####

## IoU Segmentation Computation
IoU_Original_Img = bbox.IoU_Segmentation(img_original,img_gt)
IoU_Perturbated_Img = bbox.IoU_Segmentation(img_perturbated,img_gt)

print('\nIoU Segmentation of the Original Image is:')
print(IoU_Original_Img)
print('\nIoU Segmentation of the Perturbated Image is:')
print(IoU_Perturbated_Img)
print('\nSensitivity of IoU Segmentation is:')
sensitivity = ((IoU_Perturbated_Img-IoU_Original_Img)/IoU_Original_Img)/(factor_perturbation)
print(sensitivity)
sensitivity = ((IoU_Perturbated_Img-IoU_Original_Img)/IoU_Original_Img)/(fsim_INP)
print(sensitivity)
sensitivity = ((IoU_Perturbated_Img-IoU_Original_Img)/IoU_Original_Img)/(ssim_INP)
print(sensitivity)

## Dice Coefficient
Original_dice_score = bbox.dicecoef(img_original,img_gt)
Perturbated_dice_score = bbox.dicecoef(img_perturbated,img_gt)
print('\nOriginal Dice Score is:')
print(Original_dice_score)
print('''\nPerturbated Image's Dice Score is:''')
print(Perturbated_dice_score)
print('\nSensitivity of the Dice Coefficient')
sensitivity = ((Perturbated_dice_score-Original_dice_score)/Original_dice_score)/(factor_perturbation)
print(sensitivity)
sensitivity = ((Perturbated_dice_score-Original_dice_score)/Original_dice_score)/(fsim_INP)
print(sensitivity)
sensitivity = ((Perturbated_dice_score-Original_dice_score)/Original_dice_score)/(ssim_INP)
print(sensitivity)

## Plots
cv2.imshow('Image2',img_original)
cv2.waitKey(1)

cv2.imshow('Image3',img_perturbated)
cv2.waitKey(1)

cv2.imshow('Image4',img_gt)
cv2.waitKey(0)
