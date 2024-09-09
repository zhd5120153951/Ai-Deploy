import cv2
import torch
import numpy as np


def is_poi_in_poly(pt, poly):
    """
    判断点是否在多边形内部的 pnpoly 算法
    :param pt: 点坐标 [x,y]
    :param poly: 点多边形坐标 [[x1,y1],[x2,y2],...]
    :return: 点是否在多边形之内
    """
    nvert = len(poly)
    vertx = []
    verty = []
    testx = pt[0]
    testy = pt[1]
    for item in poly:
        vertx.append(item[0])
        verty.append(item[1])

    j = nvert - 1
    res = False
    for i in range(nvert):
        if (verty[j] - verty[i]) == 0:
            j = i
            continue
        x = (vertx[j] - vertx[i]) * (testy - verty[i]) / \
            (verty[j] - verty[i]) + vertx[i]
        if ((verty[i] > testy) != (verty[j] > testy)) and (testx < x):
            res = not res
        j = i
    return res


def person_in_poly_area(xyxy):
    # 求物体框的中点
    object_x1 = int(xyxy[0])
    object_y1 = int(xyxy[1])
    object_x2 = int(xyxy[2])
    object_y2 = int(xyxy[3])
    object_w = object_x2 - object_x1
    object_h = object_y2 - object_y1
    object_cx = object_x1 + (object_w / 2)
    object_cy = object_y1 + (object_h / 2)

    return object_cx, object_cy


def cal_iou(box1, box2):  # 左下角 and 右上角
    """
    :param box1: = [xmin1, ymin1, xmax1, ymax1]
    :param box2: = [xmin2, ymin2, xmax2, ymax2]
    :return:
    """
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    # 计算每个矩形的面积
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)  # C的面积
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)  # G的面积

    # 计算相交矩形
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    area = w * h  # C∩G的面积
    iou = area / (s1 + s2 - area)
    return iou

# --------------------------------------------------------------------- #
# 只能固定1080*1920


def load_poly_area_data(data):
    '''
    获取区域的polys
    :return: 多边形的坐标 [[x1,y1],[x2,y2],……,[xn,yn],[x1,y1]] 二维数组
    '''

    pts_len = len(data['tools']['detect_area_flag']['polygon'])
    if pts_len % 2 != 0:  # 多边形坐标点必定是2的倍数
        return []

    area_poly = []
    xy_index_max = pts_len // 2
    # "x1": 402,"y1": 234,"x2": 497,"y2": 182,.....
    for i in range(0, xy_index_max):
        str_index = str(i + 1)
        h_index = 'h' + str_index
        w_index = 'w' + str_index
        one_poly = [int(data['tools']['detect_area_flag']['polygon'][w_index]), int(
            data['tools']['detect_area_flag']['polygon'][h_index])]
        area_poly.append(one_poly)
    pts = np.array(area_poly, np.int32)
    pts1 = pts[0]
    w1 = pts1[0]
    h1 = pts1[1]
    return pts, w1, h1

# -------------------------------------------------------------------- #
# 可随图片尺寸变化为变化


def get_posion(data):
    h0 = data['h0w0'][0]
    w0 = data['h0w0'][1]

    # 左上为第一个点, 顺时针
    h1 = int(data['h1']) / h0  # 左上监测区域高度距离图片顶部比例
    w1 = int(data['w1']) / w0  # 左上监测区域高度距离图片左部比例
    h2 = int(data['h2']) / h0  # 右上监测区域高度距离图片顶部比例
    w2 = int(data['w2']) / w0  # 右上监测区域高度距离图片左部比例
    h3 = int(data['h3']) / h0  # 右下监测区域高度距离图片顶部比例
    w3 = int(data['w3']) / w0  # 右下监测区域高度距离图片左部比例
    h4 = int(data['h4']) / h0  # 左下监测区域高度距离图片顶部比例
    w4 = int(data['w4']) / w0  # 左下监测区域高度距离图片左部比例
    return h1, w1, h2, w2, h3, w3, h4, w4


def get_pts(img_array_copy, h1, w1, h2, w2, h3, w3, h4, w4):
    pts = np.array([[int(img_array_copy.shape[1] * w1), int(img_array_copy.shape[0] * h1)],  # pts1
                    [int(img_array_copy.shape[1] * w2),
                     int(img_array_copy.shape[0] * h2)],  # pts2
                    [int(img_array_copy.shape[1] * w3),
                     int(img_array_copy.shape[0] * h3)],  # pts3
                    [int(img_array_copy.shape[1] * w4), int(img_array_copy.shape[0] * h4)]], np.int32)  # pts4
    return pts
# -------------------------------------------------------------------- #


def tuple_xyxy(t_list):
    t = tuple(t_list)
    return t


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(
        x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(
        x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def same(img1, img2):
    degree = 0
    H1 = cv2.calcHist([img1], [1], None, [256], [0, 256])
    H2 = cv2.calcHist([img2], [1], None, [256], [0, 256])
    for i in range(len(H1)):
        if H1[i] != H2[i]:
            degree = degree + (1 - abs(H1[i] - H2[i]) / max(H1[i], H2[i]))
        else:
            degree += 1
    degree = degree / len(H1)
    return degree
