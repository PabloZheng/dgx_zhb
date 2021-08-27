from random_walk import randomwalk, marker_only
from hough_circle import hough_circle
from canny import center_crop_canny
from image_loader import single_image, image_stream
from erosion import erosion
import cv2
import os


def transducer_eliminate(filelist, **kwargs):
    for i, filename in enumerate(filelist):
        kwargs['img_name'] = filename.split('.')[0]
        kwargs['index'] = i
        kwargs['save_path'] = '../new_save'
        img = cv2.imread(kwargs['dir_path'] + filename)
        edge = center_crop_canny(img, **kwargs)

        # EUS换能器成像-霍夫圆检测
        kwargs['save_path'] = '../new_save/hough'
        kwargs['single_image'] = True
        circles = single_image(edge, hough_circle, **kwargs)


def r_walk(filelist, **kwargs):
    for i, name in enumerate(filelist):
        read_path = kwargs['dir_path'] + f'/{name}'
        marker1_path = kwargs['save_path'] + f'/marker/marker_{name}'
        seg_path = kwargs['save_path'] + f'/segment/segment_{name}'
        if os.path.exists(marker1_path):
            print('exists:', marker1_path)
        else:
            randomwalk(read_path, marker1_path, seg_path)
            print("+1 ", marker1_path)


def centercrop(filelist, **kwargs):
    for i, name in enumerate(filelist):
        read_path = kwargs['dir_path'] + f'/{name}'
        save_path = kwargs['save_path'] + f'/{name}'
        img = cv2.imread(read_path, 0)
        (h, w) = img.shape
        img[int(3/8*h): int(5/8*h), int(3/8*w): int(5/8*w)] = 0
        cv2.imwrite(save_path, img)


if __name__ == '__main__':
    img = cv2.imread('I://dataset//EUS_bulk//train//P//EUS1_p0_img_1.jpg', 0)
    input_path = '../new_save/layer_segment/marker'
    file_list = os.listdir(input_path)
    kwargs = {'save_path': '../new_save/layer_segment/cropped', 'single_image': False, 'dir_path': input_path}
    centercrop(file_list, **kwargs)






