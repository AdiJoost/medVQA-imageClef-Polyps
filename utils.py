import numpy as np
from cv2.typing import MatLike
import cv2

#Please, do not change anything here, it might work only because of magic

def preprocess_image(rawImage: MatLike, blurredItterations: int = 3) -> MatLike:
    grayScale = cv2.cvtColor(rawImage, cv2.COLOR_BGR2GRAY)
    blurred_image = _getBlurredImage(rawImage, blurredItterations)
    binary_mask = _get_specular_highlight_region(grayScale)

    result = np.where(binary_mask[:, :, np.newaxis] == 1, blurred_image, rawImage)
    inpainted = cv2.inpaint(result, binary_mask, 10, cv2.INPAINT_TELEA)
    blackremoved = removeBlackBackground(inpainted)
    _, thresholded = cv2.threshold(grayScale, 20, 255, cv2.THRESH_BINARY_INV)
    threshold_mask = cv2.merge([thresholded] * 3)
    blackRemovedDone = np.where(threshold_mask == 255, blackremoved, inpainted)
    return blackRemovedDone, inpainted

def _get_specular_highlight_region(grayScaleImage: MatLike, threshold: int = 180) -> MatLike:
    _, thresholded = cv2.threshold(grayScaleImage, threshold, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresholded, kernel=np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(dilated)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 17: #This may be wrongly calculated
            cv2.drawContours(mask, [contour], -1, 255, -1)
    eroded = cv2.erode(mask, kernel=np.ones((3, 3), np.uint8), iterations=1)
    smoothed = cv2.GaussianBlur(eroded, (19, 19), 0)
    _, binary_mask = cv2.threshold(smoothed, 20, 1, cv2.THRESH_BINARY)
    return binary_mask

def _getBlurredImage(rawImage: MatLike, blurredItterations: int) -> MatLike:
    blurred_image = cv2.blur(rawImage, (3, 3))
    if blurredItterations > 1:
        for _ in range(blurredItterations - 1):
            blurred_image = cv2.blur(blurred_image, (3, 3))
    return blurred_image

def removeBlackBackground(rawImage: MatLike) -> MatLike:
    gray = cv2.cvtColor(rawImage, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 2, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(thresholded, kernel, iterations=2)
    distanceTransform = cv2.distanceTransform(eroded, cv2.DIST_L2, 3)
    distanceTransform = cv2.erode(distanceTransform, kernel, iterations=3)

    centerX, centerY = _getCenterPoint(distanceTransform)
    bigMaskRadius, smallMaskRadius = _getMaskRadius(distanceTransform, (centerX, centerY), 5)
    rectMask = _getRectMask(distanceTransform)
    bigCircleMask = _getCircleMask(distanceTransform, (centerX, centerY), bigMaskRadius)
    combinedSmallMask = cv2.bitwise_or(rectMask, bigCircleMask)
    smallCircleMask = _getCircleMask(distanceTransform, (centerX, centerY), smallMaskRadius)
    combined_mask = cv2.bitwise_or(rectMask, smallCircleMask)
    return cv2.inpaint(rawImage, combined_mask, 5, cv2.INPAINT_TELEA)
    replacement = cv2.inpaint(rawImage, smallCircleMask, 5, cv2.INPAINT_TELEA)
    return removeBlackPixelsBelowThresh(impainting, replacement, 25)

def removeBlackPixelsBelowThresh(impainting: MatLike, replacement: MatLike, threshold: int) -> MatLike:
    gray = cv2.cvtColor(impainting, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY_INV)
    return cv2.copyTo(replacement, thresholded, impainting)
    pass

def _getCircleMask(distanceTransform: MatLike, centerPoint: tuple, maskRadius: int) -> MatLike:
    mask_height, mask_width = distanceTransform.shape[:2]
    circle_mask = cv2.circle(np.zeros((mask_height, mask_width), dtype=np.uint8), centerPoint, maskRadius, 255, -1)
    return cv2.bitwise_not(circle_mask)

def _getRectMask(distanceTransform: MatLike) -> MatLike:
    upperLeftX = _getLeftDistance(distanceTransform)
    upperLeftY = _getUpDistance(distanceTransform)
    lowerRightY = distanceTransform.shape[0] - _getrightDistance(distanceTransform)
    lowerRightX =  distanceTransform.shape[1] - _getDownDistance(distanceTransform)
    mask_height, mask_width = distanceTransform.shape[:2]
    mask = np.zeros((mask_height, mask_width), dtype=np.uint8)

    rect_mask = cv2.rectangle(mask, (upperLeftX, upperLeftY), (lowerRightX, lowerRightY), 255, -1)
    return cv2.bitwise_not(rect_mask)

def _getMaskRadius(distanceTransform: MatLike, centerPoint: tuple, threshold: int) -> tuple:
    distanceLeftUp = _getDistanceLeftUp(distanceTransform, centerPoint, threshold)
    distanceRightUp = _getDistanceRightUp(distanceTransform, centerPoint, threshold)
    distanceRightDown = _getDistanceRightLower(distanceTransform, centerPoint, threshold)
    bigRadius = max(distanceRightUp, distanceLeftUp, distanceRightDown )
    smallRadius = min(distanceRightUp, distanceLeftUp, distanceRightDown)
    return (bigRadius, smallRadius)

def _getDistanceLeftUp(distanceTransform: MatLike, centerPoint: tuple, threshold: int) -> int:
    distance_left_upper = 0
    currentX = centerPoint[0]
    currentY = centerPoint[1]
    mostPosibleSteps = min(currentX, currentY)
    for _ in range(mostPosibleSteps - 1):
        currentX -= 1
        currentY -= 1
        if distanceTransform[currentX, currentY] < threshold:
            distance_left_upper += np.sqrt(2)
        else:
            break
    distance_left_upper = int(np.floor(distance_left_upper))
    return distance_left_upper

def _getDistanceRightUp(distanceTransform: MatLike, centerPoint: tuple, threshold: int) -> int:
    distance_right_upper = 0
    currentX = centerPoint[0]
    currentY = centerPoint[1]
    mostPosibleSteps = min(distanceTransform.shape[0] - currentX, currentY)
    for _ in range(mostPosibleSteps - 1):
        currentX += 1
        currentY -= 1
        if distanceTransform[currentX, currentY] < threshold:
            distance_right_upper += np.sqrt(2)
        else:
            break
    distance_right_upper = int(np.floor(distance_right_upper))
    return distance_right_upper

def _getDistanceRightLower(distanceTransform: MatLike, centerPoint: tuple, threshold: int) -> int:
    distance_right_lower = 0
    currentX = centerPoint[0]
    currentY = centerPoint[1]
    mostPosibleSteps = min(distanceTransform.shape[0] - currentX, distanceTransform.shape[1] - currentY)
    for _ in range(mostPosibleSteps - 1):
        currentX += 1
        currentY += 1
        if distanceTransform[currentX, currentY] < threshold:
            distance_right_lower += np.sqrt(2)
        else:
            break
    distance_right_lower = int(np.floor(distance_right_lower))
    return distance_right_lower

def _getCenterPoint(distanceTransform: MatLike) -> tuple:
    rightDistance = _getrightDistance(distanceTransform)
    leftDistance = _getLeftDistance(distanceTransform)
    upDistance = _getUpDistance(distanceTransform)
    downDistance = _getDownDistance(distanceTransform)
    x = int(((distanceTransform.shape[1] - rightDistance - leftDistance) / 2) + leftDistance)
    y = int(((distanceTransform.shape[0] - upDistance - downDistance) / 2) + upDistance)
    return (x, y) 


def _getUpDistance(distanceTransform: MatLike)-> int:
    upDistance = 0
    foundBorder = False
    for i in range(distanceTransform.shape[0]):
        for j in range(distanceTransform.shape[1]):
            if distanceTransform[i,j] == 0:
                foundBorder = True
        if foundBorder:
            break;
        upDistance += 1
    return upDistance

def _getDownDistance(distanceTransform: MatLike) -> int:
    downDistance = 0
    foundBorder = False
    for i in range(distanceTransform.shape[0] - 1, -1, -1):
        for j in range(distanceTransform.shape[1]):
            if distanceTransform[i,j] == 0:
                foundBorder = True
        if foundBorder:
            break;
        downDistance += 1
    return downDistance

def _getLeftDistance(distanceTransform: MatLike) -> int:
    leftDistance = 0
    foundBorder = False
    for j in range(distanceTransform.shape[1]):
        for i in range(distanceTransform.shape[0]):
            if distanceTransform[i,j] == 0:
                foundBorder = True
        if foundBorder:
            break;
        leftDistance += 1
    return leftDistance

def _getrightDistance(distanceTransform: MatLike) -> int:
    rightDistance = 0
    foundBorder = False
    for j in range(distanceTransform.shape[1] - 1, -1, -1):
        for i in range(distanceTransform.shape[0]):
            if distanceTransform[i,j] == 0:
                foundBorder = True
        if foundBorder:
            break;
        rightDistance += 1
    return rightDistance