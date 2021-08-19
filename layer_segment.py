from random_walk import randomwalk
from hough_circle import hough_circle
from canny import center_crop_canny
from image_loader import single_image, image_stream
import cv2
import os


if __name__ == '__main__':
    img = cv2.imread('I://dataset//EUS_bulk//train//P//EUS1_p0_img_1.jpg', 0)
    dir_name = 'I://dataset//EUS_bulk//train//P//'
    # file_list = image_stream(dir_name, center_crop_canny, **kwargs)
    file_list = os.listdir(dir_name)
    kwargs = {'save_path': '../new_save', 'single_image': False}

    for i, filename in enumerate(file_list):
        kwargs['img_name'] = filename.split('.')[0]
        kwargs['index'] = i
        kwargs['save_path'] = '../new_save'
        img = cv2.imread(dir_name + filename)
        edge = center_crop_canny(img, **kwargs)
        kwargs['save_path'] = '../new_save/hough'
        kwargs['single_image'] = True
        circles = single_image(edge, hough_circle, **kwargs)
        
        # if i >=100:
        #     break