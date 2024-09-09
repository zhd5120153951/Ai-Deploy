import copy

import cv2
import numpy as np
import torch

from ai_common.log_util import logger
from ai_common.np_util import person_in_poly_area, is_poi_in_poly, tuple_xyxy, load_poly_area_data, xyxy2xywh, \
    xywh2xyxy
from ai_common.util.general import scale_coords, non_max_suppression

from analysis.extract_param import param

from ai_common.util.plots import plot_one_box


def second_classifier(data, x, model, img, im0):
    params = param(data)
    names = model.module.names if hasattr(model, 'module') else model.names
    # applies a second stage classifier to yolo outputs
    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    smoke_one_hand = []
    for i, d in enumerate(x):  # per hand
        if d is not None and len(d):
            d = d.clone()
            hand = d.clone()
            # Reshape and pad cutouts  扩大手部图像
            b = xyxy2xywh(d[:, :4])  # boxes
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # rectangle to square
            b[:, 2:] = b[:, 2:] * 2 + 60  # pad2
            d[:, :4] = xywh2xyxy(b).long()
            pad_x = d[:, :4] - hand[:, :4]

            # Rescale boxes from img_size to im0 size 将裁剪后的图像恢复到原来的尺寸
            scale_coords(img.shape[2:], d[:, :4], im0[i].shape)
            shape = img.shape[2:]
            shape0 = im0[i].shape
            pad_x[:, 0] = pad_x[:, 0] * (im0[i].shape[0] / shape[0])
            pad_x[:, 1] = pad_x[:, 1] * (im0[i].shape[1] / shape[1])

            # Classes
            pred_cls1 = d[:, 5].long()
            ims = []
            ims_size = []
            for j, a in enumerate(d):  # per item
                print(d)
                print(a)
                cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]
                im_size = cutout.shape
                ims_size.append(im_size)
                im = cv2.resize(cutout, (224, 224))  # BGR

                gamma = 1.3

                out = 255 * (im / 255) ** gamma

                im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
                im /= 255.0  # 0 - 255 to 0.0 - 1.0
                ims.append(im)

            pred_cls2 = model(torch.Tensor(ims).to(d.device))[0]  # classifier prediction
            pred_cls2 = non_max_suppression(pred_cls2, params.conf_thres_smoke, params.iou_thres_smoke)
            c1 = []
            for j, a in enumerate(d):
                if (pred_cls2[j] is not None and len(pred_cls2[j])):
                    num = 0
                    for *xyxy, conf, cls in pred_cls2[j]:
                        pred_cls2[j][num][0] = pred_cls2[j][num][0] * ims_size[j][0] / 224
                        pred_cls2[j][num][1] = pred_cls2[j][num][1] * ims_size[j][1] / 224
                        pred_cls2[j][num][2] = pred_cls2[j][num][2] * ims_size[j][0] / 224
                        pred_cls2[j][num][3] = pred_cls2[j][num][3] * ims_size[j][1] / 224
                        c1 = (int(a[0]), int(a[1]), int(a[0]), int(a[1]))
                        c1 = list(c1)
                        c1 = torch.Tensor(c1).to(d.device)
                        pad_zero = torch.Tensor([0, 0]).to(d.device)
                        c1 = torch.cat((c1, pad_zero), dim=0)
                        pred_cls2[j][num] = c1 + pred_cls2[j][num]
                        num = num + 1
            smoke_one_hand.append(pred_cls2)

    return smoke_one_hand


def process_result_hand(MODEL_smoke, pred_hand, names_hand, names_smoke, data, img, img_array):
    params = param(data)
    img_array_copy = img_array.copy()
    hand = {}
    s = ''
    pts, w1, h1 = load_poly_area_data(data)
    pred_hand0 = copy.deepcopy(pred_hand)


    for i, det in enumerate(pred_hand):
        s += '%g x %g ' % img.shape[2:]  # logger.info string, 取第三和第四维，对应宽和高 512 x 640

        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_array_copy.shape).round()

            # logger.info results
            for c in det[:,-1].unique():
                n = (det[:, -1] == c).sum()
                s += f"{n} {names_hand[int(c)]}{'s' * (n > 1)}, "

            # judge results
            for *xyxy, conf, cls in reversed(det):
                hand_list = ['hand']
                c = int(cls)  # integer class 1， 0
                xyxy = [xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()]

                # label
                if params.hide_labels:
                    label = None
                elif params.hide_conf:
                    label = names_hand[c]
                else:
                    label = names_hand[c] + "%.2f" % conf

                if params.detect_area_flag:
                    # 求物体框的中心点
                    object_cx, object_cy = person_in_poly_area(xyxy)
                    # 判断中心点是否在检测框内部
                    if not is_poi_in_poly([object_cx, object_cy], pts):
                        # 不在感兴趣的框内，则继续判断下一个物体。
                        continue
                    if names_hand[c] in hand_list:
                        if data['tools']['Target_filter']['filter_switch'] == True:
                            if (int(xyxy[2]) - int(xyxy[0])) * (int(xyxy[3]) - int(xyxy[1])) > int(data['min_people_xyxy']):
                                hand[tuple_xyxy(xyxy)] = label
                        else:
                            hand[tuple_xyxy(xyxy)] = label
                else:
                    if names_hand[c] in hand_list:
                        if data['tools']['Target_filter']['filter_switch'] == True:
                            if (int(xyxy[2]) - int(xyxy[0])) * (int(xyxy[3]) - int(xyxy[1])) > int(
                                    data['min_people_xyxy']):
                                hand[tuple_xyxy(xyxy)] = label
                        else:
                            hand[tuple_xyxy(xyxy)] = label
                            pass
                        pass
                    pass
                pass
            pass
        pass
    logger.info(f'hand detect result: {s}Done.')
    # for h in hand:
    #     plot_one_box(h, img_array_copy, label=hand[h], color=(255, 0, 0), line_thickness=params.line_thickness)
    # cv2.imshow('hand_detect', img_array_copy)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    smoke = {}

    pred0 = second_classifier(data, pred_hand0, MODEL_smoke, img, img_array_copy)
    if pred0 is not None and len(pred0):
        pred0 = pred0[0]
    else:
        return hand, img_array_copy

    ss = ''
    for i, det in enumerate(pred0):
        ss += '%g x %g ' % img.shape[2:]  # print string, 取第三和第四维，对应宽和高 512 x 640d()

        # logger.info results
        for cs in det[:, -1].unique():
            ns = (det[:, -1] == cs).sum()
            ss += f"{ns} {names_smoke[int(cs)]}{'s' * (ns > 1)}, "

        # Write results
        for *xyxy, conf, cls in det:
            cs = int(cls)
            xyxy = [xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()]

            # label
            if params.hide_labels:
                label = None
            elif params.hide_conf:
                label = names_smoke[cs]
            else:
                label = names_smoke[cs] + "%.2f" % conf

            smoke_list = ['smoke']
            if params.detect_area_flag:
                # 求物体框的中心点
                object_cx, object_cy = person_in_poly_area(xyxy)
                # 判断中心点是否在检测框内部
                if not is_poi_in_poly([object_cx, object_cy], pts):
                    # 不在感兴趣的框内，则继续判断下一个物体。
                    continue
                if names_smoke[cs] in smoke_list:
                    if data['tools']['Target_filter']['filter_switch'] == True:
                        if (int(xyxy[2]) - int(xyxy[0])) * (int(xyxy[3]) - int(xyxy[1])) > int(data['min_people_xyxy']):
                            smoke[tuple_xyxy(xyxy)] = label
                    else:
                        smoke[tuple_xyxy(xyxy)] = label
            else:
                if names_smoke[cs] in smoke_list:
                    if data['tools']['Target_filter']['filter_switch'] == True:
                        if (int(xyxy[2]) - int(xyxy[0])) * (int(xyxy[3]) - int(xyxy[1])) > int(data['min_people_xyxy']):
                            smoke[tuple_xyxy(xyxy)] = label
                    else:
                        smoke[tuple_xyxy(xyxy)] = label

    logger.info(f'smoke detect result: {ss}Done.')


    return hand, smoke, img_array_copy
