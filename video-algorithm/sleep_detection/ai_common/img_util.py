import copy

import cv2
import base64
import numpy as np
import torch


def letterbox(img0, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    img = copy.copy(img0)
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


# 传入应该为BGR格式的np数组，输出为jpg格式的base64编码字符串。
def img_to_base64(image_np):
    img = cv2.imencode(".jpg", image_np)[1]
    img64 = str(base64.b64encode(img))[2:-1]
    return img64


def path2base64(path):
    with open(path, "rb") as f:
        byte_data = f.read()
    base64_str = base64.b64encode(byte_data).decode("ascii")
    return base64_str


# 传入为RGB格式下的base64，传出为RGB格式的numpy矩阵
def base64_to_img(base64_str, img_size=640):
    # 将base64转换为二进制
    byte_data = base64.b64decode(base64_str)
    # 二进制转换为一维数组
    encode_image = np.asarray(bytearray(byte_data), dtype="uint8")
    # 用cv2解码为三通道矩阵
    img_array = cv2.imdecode(encode_image, cv2.IMREAD_COLOR)
    h0w0 = img_array.shape[0:2]
    # 将图像尺寸调整为32像素倍数的矩形，输入和输出图像格式均为BGR。不知道用RGB是否可以。
    letter_img = letterbox(img_array, new_shape=img_size)[0]
    # BGR2RGB
    img_RGB = letter_img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    # 函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快。
    img_RGB = np.ascontiguousarray(img_RGB)
    return img_RGB, h0w0, img_array


def similarity(img, nparray):
    im0s = img.copy()
    im0s = cv2.cvtColor(im0s.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
    if nparray is None:
        H1 = cv2.calcHist([im0s], [1], None, [256], [0, 256])
        degree = 0
        return degree, H1
    else:
        degree = 0
        H1 = cv2.calcHist([im0s], [1], None, [256], [0, 256])
        for i in range(len(H1)):
            if H1[i] != nparray[i]:
                degree = degree + (1 - abs(H1[i] - nparray[i]) / max(H1[i], nparray[i]))
            else:
                degree += 1
        degree = degree / len(H1)
        return degree, H1


def detection_region(img_array_copy, w1, h1, pts):
    cv2.putText(img_array_copy, "Detection_Region", (int(img_array_copy.shape[1] * w1 - 5), int(img_array_copy.shape[0] * h1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2, cv2.LINE_AA)
    img_array_copy = cv2.polylines(img_array_copy, [pts], True, (255, 255, 0), 2)  # 画感兴趣区域
    return img_array_copy


def preprocess_tensor(img, device, half):
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    return img


def preprocess_np(img):
    img = img/255.0  # 0 - 255 to 0.0 - 1.0
    img = np.ascontiguousarray(img)
    if img.ndim == 3:
        img = np.expand_dims(img, 0)
    img = np.array(img, dtype=np.float32)

    return img


