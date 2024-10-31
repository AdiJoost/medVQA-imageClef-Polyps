import numpy as np
from numpy import MatLike
import cv2

def preprocess_image(rawImage: MatLike, blurredItterations: int = 3) -> MatLike:
    grayScale = cv2.cvtColor(rawImage, cv2.COLOR_BGR2GRAY)
    blurred_image = _getBlurredImage(rawImage, blurredItterations)
    binary_mask = _get_specular_highlight_region(grayScale)

    result = np.where(binary_mask[:, :, np.newaxis] == 1, blurred_image, rawImage)
    inpainted = cv2.inpaint(result, binary_mask, 10, cv2.INPAINT_TELEA)
    return inpainted

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