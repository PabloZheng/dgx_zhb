import cv2 as cv
import numpy as np


def erode_demo(image):
    print(image.shape)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV|cv.THRESH_OTSU)
    cv.imshow("binary", binary)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    dst = cv.erode(binary, kernel)
    cv.imshow("erode_demo", dst)

# original image
# src = cv.imread("I://dataset//EUS_bulk//train//P//EUS1_p0_img_1.jpg")

# marker image
# src = cv.imread("I://dataset//marker//marker_EUS1_p47_img_1.jpg")

# segment image
src = cv.imread("I:\dataset\segment\segment_EUS1_p1_img_1.jpg")

cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
erode_demo(src)

cv.waitKey(0)
cv.destroyAllWindows()
