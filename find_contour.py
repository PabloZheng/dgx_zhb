import cv2
import numpy as np
import heapq
import os
import shutil

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


def erosion(image):
    print(image.shape)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # cv2.imshow("binary", binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dst = cv2.erode(binary, kernel)
    # cv2.imshow("erode_demo", dst)
    # cv2.waitKey()
    # cv.imwrite('fil', dst)
    return dst


def get_contour(img):
    ''' 获取连通域

    :param img:
    :return: 最大联通域
    '''

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TRUNC)
    # cv2.imshow('thresholded',img_bin)
    # cv2.waitKey()
    contours, hierachy = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    return img_gray, contours[0]


def capture_layers(img_src, contours, max_area_num=3, **kwargs):
    '''

    :param img_src: input image
    :param contours: contours set
    :param max_area_num: the top n areas of contours you want to keep
    :param kwargs: saving path params
    :return: Empty
    '''
    area_dic = {}
    for i, contour in enumerate(contours):
        # print(f'shape:{contour.shape}, coordinate:{contour}\n')
        area = cv2.contourArea(contour)
        area_dic[i] = area
        print('contour area:', area)

    area_values = list(area_dic.values())
    max_num_index_list = map(area_values.index, heapq.nlargest(max_area_num, area_values))
    index = list(max_num_index_list)
    if len(index) < max_area_num:
        max_area_num = len(index)
    mask_ground = np.zeros(img_src.shape, np.uint8)
    for i in range(0, max_area_num):
        print(index[i], ':', area_dic[index[i]])
        contour_draw = contours[index[i]]
        cv2.drawContours(mask_ground, contour_draw, -1, (225, 0, 255), 3)
        cv2.fillPoly(mask_ground, [contour_draw], color=(255, 0, 0))

        # x, y, w, h = cv2.boundingRect(contours[index[i]])
        # cv2.rectangle(img_src, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.imshow('rectangle', mask_ground)
        save_path = kwargs['save_path']
        file_name = kwargs['file_name']
        # cv2.imwrite(f'{save_path}/{file_name}', mask_ground)
        kwargs['save_path'] = '../new_save/layer_segment/rectangle_mask'
        eus_img = kwargs['eus_name']
        img_src = cv2.imread(f'../new_save/layer_segment/EUS_images/{eus_img}')
        draw_rectangle_mask(img_src, contour_draw, **kwargs)

        # cv2.waitKey()


def find_contour_main(img_src, **kwargs):
    # img_src = cv2.imread('../new_save/layer_segment/marker/marker_P_p4_img21.jpg')
    dst = erosion(img_src)
    contours = find_contour(dst)
    capture_layers(img_src, contours, 3, **kwargs)


def draw_rectangle_mask(img_src, contour_draw, **kwargs):
    contour_draw_list = list(contour_draw)
    points_number = len(contour_draw_list)
    # for i in range(1):
    center_index = np.random.randint(0, points_number)
    center = contour_draw_list[center_index][0]
    img_src[center[0]-30:center[0]+30, center[1]-30:center[1]+30] = 0
    save_path = kwargs['save_path']
    # file_name = kwargs['file_name']
    eus_name = kwargs['eus_name']
    eus_name = eus_name.split('.jpg')[0]
    save_path = f'{save_path}/{eus_name}.jpg'
    if os.path.exists(save_path):
        save_path = f'{save_path}/{eus_name}_{np.random.randint(1,100)}.jpg'
    else:
        pass
    cv2.imwrite(save_path, img_src)


def move_eus():
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
            for k in range(len(file_list)):
                shutil.copy(third_path+f'/{file_list[k]}', f'../new_save/layer_segment/EUS_images/{file_list[k]}')
            kwargs = {'save_path': '../new_save', 'single_image': False, 'dir_path': dir_name}
            # transducer_eliminate(file_list, **kwargs)
            kwargs['save_path'] = '../new_save/layer_segment'


if __name__ == '__main__':
    input_folder = '../new_save/layer_segment/marker'
    # move_eus()
    file_list = os.listdir(input_folder)
    kwargs = {}
    kwargs['save_path'] = '../new_save/layer_segment/layer_mask'
    for i, file_name in enumerate(file_list):
        img_src = cv2.imread(input_folder + f'/{file_name}')
        kwargs['file_name'] = file_name
        kwargs['eus_name'] = file_name.split('marker_')[1]
        find_contour_main(img_src, **kwargs)
