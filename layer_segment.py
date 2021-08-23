from random_walk import randomwalk, marker_only
from hough_circle import hough_circle
from canny import center_crop_canny
from image_loader import single_image, image_stream
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
            pass
        else:
            randomwalk(read_path, marker1_path, seg_path)
            print("+1 ", marker1_path)


if __name__ == '__main__':
    # img = cv2.imread('I://dataset//EUS_bulk//train//P//EUS1_p0_img_1.jpg', 0)
    first_path = 'I:/dataset/EUS_bulk'
    _first_path = os.listdir('I:/dataset/EUS_bulk')
    for i in range(len(_first_path)):
        second_path = first_path + f'/{_first_path[i]}'
        _second_path = os.listdir(second_path)
        for j in range(len(_second_path)):
            third_path = second_path + f'/{_second_path[j]}'
            # dir_name = 'I://dataset//EUS_bulk//train//P//'
            dir_name = third_path
            # file_list = image_stream(dir_name, center_crop_canny, **kwargs)
            file_list = os.listdir(dir_name)
            kwargs = {'save_path': '../new_save', 'single_image': False, 'dir_path': dir_name}
            # transducer_eliminate(file_list, **kwargs)
            kwargs['save_path'] = '../new_save/layer_segment'
            r_walk(file_list, **kwargs)

    # file_list = image_stream(dir_name, create_marker, **kwargs)





