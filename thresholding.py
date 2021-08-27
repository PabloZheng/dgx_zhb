import cv2
import numpy as np
import matplotlib.pylab  as plt
from steger_line import *


def threshold(img):
    '''

    :param img:
    :return:
    '''

    ret2, thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
    # cv2.imshow('ret', thresh2)
    # cv2.waitKey()
    plt.imshow(thresh3)
    plt.show()
    return thresh3


def find_contour(img):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    # binary = binary*(-1) + 255

    # cv2.imshow("img", binary)
    # cv2.waitKey(0)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (225, 0, 255), 3)

    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    return contours


if __name__ == '__main__':
    img = cv2.imread('./154.jpg')
    thresh = threshold(img)
    cv2.resize(img, (200, 200))
    edges = cv2.Canny(thresh, 20, 200)
    plt.imshow(edges)
    plt.show()

    # contours = find_contour(thresh)
    # mask_ground = np.zeros(img.shape, np.uint8)
    # cv2.drawContours(mask_ground, contours, -1, (225, 0, 255), 3)
    # cv2.fillPoly(mask_ground, [contours], color=(255, 0, 0))
    # cv2.imshow('s', mask_ground)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, (640, 480))
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # compute derivative
    # dx, dy, dxx, dyy, dxy = computeDerivative(gray_img, 1.1, 1.1)
    # normal, phase = computeMagnitude(dxx, dyy)
    # pt, dir, val = computeHessian(dx, dy, dxx, dyy, dxy)
    #
    # # take the first n max value
    # nMax = 10000
    # idx = np.argsort(val)
    # idx = idx[::-1][:nMax]
    # # plot resulting point
    # for i in range(0, len(idx)):
    #     img = cv2.circle(img, (pt[idx[i]][0], pt[idx[i]][1]), 1, (255, 0, 0), 1)
    # plt.imshow(phase)
    # plt.show()
