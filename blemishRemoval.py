"""
Author: Piotr Książak
blemishRemoval.py
"""

import numpy as np
import cv2 as cv

ESC = 27
BACKSPACE = 8

# Program works in a specific way:
# 1) First, performing inital blurring to the original image
#       for the purpose of finding blemishes
# 2) Next, by clicking on the blemish, I'm enlarging the original
#       image with border, so I can choose the blemish near the corner,
#       and perform later searching for a patch also near the corner
# 3) Next, applying Scharr derivative on the box with the blemish
# 4) Next, detecting the blemish inside this box, becuase I don't know
#       how big is the blemish inside the box or if the user clicked exactly
#       on the center
# 5) Next, when the blemish is detected enclosed by the proper circle,
#       I'm saving this circle as the mask, so I can use it to find the 
#       mean value of the pixels around the blemish, and to paste in the next
#       steps the chosen patch with this circular mask
# 6) Next, searching for a patch around the blemish box in specific number of
#       points, but how I decide what patch to choose? I'm selecting the patch
#       which has the mean value of pixels the nearest to the value of pixels
#       around the blemish in the selected blemish box
# 7) Finally, performing seamless cloning on the image with border,
#       and copying this image but without border to the original 'img'

def onMouse(event, x, y, flags, userdata):
    global img, imgGray, imgClone
    if event == cv.EVENT_LBUTTONDOWN:
        # UNDO option
        # copying the image for undo option
        imgClone = img.copy()
        
        # the coordinates of the selected point 
        pt = (x, y)
        
        # size of the square with the blemish
        size = 35
        halfSize = int((size - 1) / 2)
        
        ##################################
        ### Adding border ################
        # enlarging the image with the border
        # to be able to choose points near the border
        bordSize = size + halfSize
        imgGrayWithBorder = cv.copyMakeBorder(imgGray, bordSize, bordSize, 
                                          bordSize, bordSize,
                                          cv.BORDER_REFLECT_101)
        imgWithBorder = cv.copyMakeBorder(img, bordSize, bordSize,
                                          bordSize, bordSize,
                                          cv.BORDER_REFLECT_101)
        
        # moving (x, y), because now working on the image with border
        pt2 = (x + bordSize, y + bordSize)
        blemRect = imgGrayWithBorder[pt2[1] - halfSize : pt2[1] + halfSize + 1,
                                        pt2[0] - halfSize : pt2[0] + halfSize + 1]
        
        ##################################
        ### Computing the Scharr derivative
        ### of the blemish box
        #ksize = cv.FILTER_SCHARR
        sobelx = cv.Scharr(blemRect, cv.CV_32F, 1, 0)
        sobely = cv.Scharr(blemRect, cv.CV_32F, 0, 1)
        sobelx = cv.convertScaleAbs(sobelx)
        sobely = cv.convertScaleAbs(sobely)
        blemRectGradient = cv.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
        retval, blemRectThresh = cv.threshold(blemRectGradient,
                                         50, 255, cv.THRESH_BINARY)
        
        ###################################
        ### Detecting the blemish
        contours, hierarchy = cv.findContours(blemRectThresh,
                                              cv.RETR_EXTERNAL,
                                              cv.CHAIN_APPROX_SIMPLE)
        max_area, max_id, area = 0, 0, 0
        for i in range(len(contours)):
            area = cv.contourArea(contours[i])
            if max_area < area:
                max_id = i
                max_area = area
        blemRectThresh = cv.drawContours(blemRectThresh, contours,
                                            max_id, (100, 0, 0), 1)
        
        center, radius = cv.minEnclosingCircle(contours[max_id])
        center = (int(center[0]), int(center[1]))
        radius = int(radius)
        
        # mask enclosing the detected blemish to find the mean in the box
        mask1 = np.zeros(blemRectThresh.shape, dtype=blemRectThresh.dtype)
        mask1 = cv.circle(mask1, center, radius, (255, 255, 255), -1)
        
        # mask enclosing the detected blemish enlarged to perform seamless cloning
        mask2 = np.zeros(blemRectThresh.shape, dtype=blemRectThresh.dtype)
        mask2 = cv.circle(mask2, center, halfSize, (255, 255, 255), -1)
        
        # What is the mean of the area around the blemish?
        mask1Not = cv.bitwise_not(mask1)
        meanBlem = cv.mean(blemRect, mask1Not)
        
        #############################
        ### Searching for a patch ### 
        nearestMean, difference, k = 255, 0, 5
        move = int(size / ((k - 1) / 2.0))
        start_x = pt2[0] - size
        start_y = pt2[1] - size
        patch_x, patch_y = 0, 0
        for i in range(k):
            for j in range(k):
                moved_x = start_x + j * move
                moved_y = start_y + i * move
                if i == 1 and j == 1:
                    continue
                # searching for the square with the nearest mean
                #  to the mean around the blemish
                square = imgGrayWithBorder[moved_y - halfSize : moved_y - halfSize + size,
                                           moved_x - halfSize : moved_x - halfSize + size]
                theMean = cv.mean(square, mask1)
                difference = abs(meanBlem[0] - theMean[0])
                if difference < nearestMean:
                    nearestMean = difference
                    patch_x = moved_x - halfSize
                    patch_y = moved_y - halfSize
                    
        patch = imgWithBorder[patch_y : patch_y + size,
                              patch_x : patch_x + size]
        
        ########################
        ### Seamless cloning ###
        imgWithBorder = cv.seamlessClone(patch, imgWithBorder, mask2, pt2, cv.NORMAL_CLONE)
        # copying the modified image
        img = imgWithBorder[bordSize : img.shape[0] + bordSize,
                            bordSize : img.shape[1] + bordSize]

def main():
    global img, imgGray, imgClone
    
    winName = "Blemish Removal"
    fileName = "blemish.png"
    img = cv.imread(fileName, cv.IMREAD_COLOR)
    if img is None:
        print("Error! Couldn't load an image!")
        return -1
    
    # initial modifications of the image
    imgBlur = cv.GaussianBlur(img, (5, 5), 0, 0)
    imgBlur = cv.medianBlur(imgBlur, 3)
    imgGray = cv.cvtColor(imgBlur, cv.COLOR_BGR2GRAY)
    
    # displaying main window for mouse callback
    cv.namedWindow(winName, cv.WINDOW_AUTOSIZE)
    cv.setMouseCallback(winName, onMouse)
    
    keyPressed = 0
    while keyPressed != ESC:
        cv.imshow(winName, img)
        keyPressed = cv.waitKey(1) & 0xFF
        
        # the undo option
        if keyPressed == BACKSPACE:
            img = imgClone
    
    # ending
    cv.destroyAllWindows()

if (__name__ == "__main__"):
    main()
