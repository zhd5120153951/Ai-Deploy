import math


def bbx_ratios(xyxy):
    xmin1,ymin1,xmax2,ymax2=xyxy
    ratio=(ymax2-ymin1)/(xmax2-xmin1)
    return ratio


def angle(head,person):
    xh1,yh1,xh2,yh2=head
    xp1,yp1,xp2,yp2=person
    person_w=xp2-xp1
    person_h=yp2-yp1
    head_w=xh2-xh1
    head_h=yh2-yh1
    head_cx=xh1+(head_w/2)
    head_cy=yh1+(head_h/2)
    person_cx=xp1+(person_w/2)
    person_cy=yp1+(person_h/2)
    if head_cy<person_cy:
        return 0
    else:
       k=(person_cy-head_cy)/(person_cx-head_cx)
       angles=math.atan(k)
       angles=math.degrees(angles)
       return abs(angles)


def cal_iou_head(box1, box2):  # head person
    """
    :param box1: = [xmin1, ymin1, xmax1, ymax1]
    :param box2: = [xmin2, ymin2, xmax2, ymax2]
    :return:
    """
    xmin1, ymin1, xmax1, ymax1 = box1  # head
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
    iou = area / s1
    return iou


