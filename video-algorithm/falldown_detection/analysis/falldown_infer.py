from analysis.analysis_utils import bbx_ratios

from ai_common import setting
from ai_common.np_util import cal_iou, tuple_xyxy
from ai_common.util.plots import plot_one_box, colors
from analysis.analysis_utils import cal_iou_head

from analysis.analysis_utils import angle


def falldown_inferrnce(data, names, c, head_helmet_list, conf, h_h_conf, xyxy, person_list, img_array_copy, cal_ious, standbbx_ratio, d_frame, stand_angle, flag):
    xyxy = tuple_xyxy(xyxy)
    if names[c] in head_helmet_list and conf > h_h_conf:
        setting.head1[data['k8sName']].append(xyxy)
    if names[c] in person_list:
        # 第一帧图片
        if setting.frame_t[data['k8sName']] == 0:
            setting.person[data['k8sName']][xyxy] = 0  # 人的坐标{xyxy:0}
            bbx = bbx_ratios(xyxy)  # 计算坐标的比例
            setting.person_bbxr[data['k8sName']][xyxy] = bbx  # {xyxy:坐标比例}
            setting.person_frame[data['k8sName']][xyxy] = 0  # 第一帧
        else:  # 正常帧（及大于第一帧）
            new_person = 0  # new_person 相用来记录当前帧与上一帧是否有对应的人，充当媒介。无则为0，有则为当前帧坐标框坐标
            bbx_r = bbx_ratios(xyxy)
            plot_one_box(xyxy, img_array_copy, label=str(bbx_r), color=colors(c, True), line_thickness=1)
            for p1, flag1 in setting.person[data['k8sName']].items():  # 上一帧图片中的人坐标
                if p1 is not None:
                    # ----- 追踪上一帧图片中的相同目标，如果没有相同目标，结束本次循环 ----- #
                    ious = cal_iou(xyxy, p1)
                    if ious > cal_ious:
                        print('ious ', ious)
                        new_person = p1
                        setting.person[p1][data['k8sName']] = p1
                        # ------------------------------------ #
                        # ----- 如果比例框【持续】小于阈值，则判断摔倒 ----- #
                        if bbx_r < standbbx_ratio:  # 比例小于阈值
                            if setting.person_frame[data['k8sName']][p1] >= d_frame:  # 比例框持续小于阈值
                                one_head = 0  # one_head 用来记录这个人是否检测到头，充当媒介，无则为0，有则为1.
                                # ---- 判断是否同时检测到身体和头。有头根据比例框和头的角度判断、无头根据比例狂判断 ---- #
                                for head_ in setting.head2[data['k8sName']]:
                                    print('head_person ', cal_iou_head(head_, p1))
                                    if cal_iou_head(head_, p1) > 0.95:
                                        one_head = 1
                                        angles = angle(head_, p1)
                                        print('angle', angles)
                                        if angles < stand_angle:
                                            plot_one_box(xyxy, img_array_copy, label='falling', color=colors(c, True))
                                            flag = False
                                            print('this person is falling!')
                                            break
                                        else:
                                            setting.person_frame[data['k8sName']][p1] = 0
                                            break
                                    else:
                                        continue
                                if one_head == 0:  # 检测不到头的时候，只用比例框判定
                                    plot_one_box(xyxy, img_array_copy, label='falling', color=colors(c, True))
                                    flag = False
                                    print('this person is falling!')
                                    break
                            else:
                                setting.person_frame[data['k8sName']][p1] += 1  # 持续帧数加一
                                break
                        else:
                            setting.person_frame[data['k8sName']][p1] = 0  # 没有持续增大，持续帧数重置为0
                            break
                    else:
                        continue

            if new_person == 0:  # 如果在基础框中没有找到对应的人
                setting.person_new[data['k8sName']][xyxy] = 0
                setting.person_newbbx[data['k8sName']][xyxy] = bbx_r
            else:  # 更新坐标和比例
                setting.person[data['k8sName']].pop(new_person)
                frame = setting.person_frame[data['k8sName']].pop(new_person)
                b = setting.person_bbxr[data['k8sName']].pop(new_person)
                setting.person[data['k8sName']][xyxy] = 1
                setting.person_bbxr[data['k8sName']][xyxy] = b
                setting.person_frame[data['k8sName']][xyxy] = frame

    return flag