import copy
import cv2
import numpy as np
import torch

from models.LPRNET import CHARS
from ai_common.util.general import scale_coords, xyxy2xywh, xywh2xyxy

def transform( img):
    img = img.astype('float32')
    img -= 127.5
    img *= 0.0078125
    img = np.transpose(img, (2, 0, 1)) # 转置矩阵

    return img


def ocr_det(x, model, img, im0):
    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    plat_num=0
    for i, d in enumerate(x):  # per image
        if d is not None and len(d):
            d = copy.deepcopy(d)
            # --------1.还原预测框reshape
            b = xyxy2xywh(d[:, :4])  # boxes
            d[:, :4] = xywh2xyxy(b).long()
            scale_coords(img.shape[2:], d[:, :4], im0[i].shape)

            # --------2.裁剪目标
            cut_img_list = [] #
            for j, a in enumerate(d):  # per item
                # --------2.1对cut img 进行模型LPR输入前预处理
                cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])] # 裁剪出目标图像
                im = cv2.resize(cutout, (94, 24))  # BGR/调整成LPR的输入尺寸
                im = transform(im)
                cut_img_list.append(im)

            # ---------3.LPR模型处理图片
            '''
            序列（列表）中每个位置对所有类别的概率,网络输出[68, 18]。68类
            68代表字典中字符总个数每个类别相应的概率
            18个预测序列长度
            '''
            preds = model(torch.Tensor(cut_img_list).to(d.device))
            # detach()通俗来讲就是返回一个和之前一摸一样的Tensor数据，并共用一个内存地址
            prebs = preds.cpu().detach().numpy()

            # --------4.后处理
            # 对识别结果进行CTC后处理：删除序列中空白位置的字符，删除重复元素的字符
            preb_labels = list()
            for w in range(prebs.shape[0]): # batch

                preb = prebs[w, :, :]
                preb_label = list()
                for j in range(preb.shape[1]):
                    preb_label.append(np.argmax(preb[:, j], axis=0)) # 序列中每个位置的最大概率对应的类别.长度18的序列（列表）

                no_repeat_blank_label = list()
                pre_c = preb_label[0]

                if pre_c != len(CHARS) - 1: # '-'，代表空字符
                    no_repeat_blank_label.append(pre_c)
                for c in preb_label:  # dropout repeate label and blank label
                    if (pre_c == c) or (c == len(CHARS) - 1):
                        if c == len(CHARS) - 1:
                            pre_c = c
                        continue
                    no_repeat_blank_label.append(c)
                    pre_c = c
                preb_labels.append(no_repeat_blank_label)

            plat_num = np.array(preb_labels)
    return x, plat_num