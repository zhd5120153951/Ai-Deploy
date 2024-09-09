import copy
from ai_common.np_util import *
from ai_common.util.general import *
from ai_common.log_util import logger


def process_result_person(pred_person, names_person, data, img, img_array, params):

    img_array_copy = copy.deepcopy(img_array)
    person = {}
    head_helmet = {}
    s = ''
    pts, w1, h1 = load_poly_area_data(data)
    for i, det in enumerate(pred_person):  # detections per image
        det = copy.deepcopy(det)
        s += '%g x %g ' % img.shape[2:]  # print string, 取第三和第四维，对应宽和高 512 x 640

        if det is not None:  # Tensor:(6, 11)
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_array_copy.shape).round()

            # # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # 检测到目标的总和  # tensor([ True, False, False])
                s += f"{n} {names_person[int(c)]}{'s' * (n > 1)}, "

            for *xyxy, conf, cls in reversed(det):
                helmet_head_list = ['helmet', 'head']
                person_list = ['person']
                c = int(cls)  # integer class 1， 0

                # label
                if params.hide_labels:
                    label = None
                elif params.hide_conf:
                    label = names_person[c]
                else:
                    label = names_person[c] + "%.2f" % conf

                xyxy = [xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()]

                # 指定了检测区域
                if params.detect_area_flag:
                    # 求物体框的中心点
                    object_cx, object_cy = person_in_poly_area(xyxy)
                    # 判断中心点是否在检测框内部
                    if not is_poi_in_poly([object_cx, object_cy], pts):
                        # 不在感兴趣的框内，则继续判断下一个物体。
                        continue
                    if data['tools']['Target_filter']['filter_switch'] == True:
                        if names_person[c] in person_list:
                            if params.max_person_xyxy >= (int(xyxy[2]) - int(xyxy[0])) * (int(xyxy[3]) - int(xyxy[1])) >= params.min_people_xyxy:
                                person[tuple_xyxy(xyxy)] = label
                        if names_person[c] in helmet_head_list:
                            if params.max_head_xyxy >= (int(xyxy[2]) - int(xyxy[0])) * (int(xyxy[3]) - int(xyxy[1])) >= params.min_head_xyxy:
                                head_helmet[tuple_xyxy(xyxy)] = label
                    else:
                        if names_person[c] in person_list:
                            person[tuple_xyxy(xyxy)] = label
                        if names_person[c] in helmet_head_list:
                            head_helmet[tuple_xyxy(xyxy)] = label
                else:
                    if data['tools']['Target_filter']['filter_switch'] == True:
                        if names_person[c] in person_list:
                            if params.max_person_xyxy >= (int(xyxy[2]) - int(xyxy[0])) * (int(xyxy[3]) - int(xyxy[1])) >= params.min_people_xyxy:
                                person[tuple_xyxy(xyxy)] = label
                        if names_person[c] in helmet_head_list:
                            if params.max_head_xyxy >= (int(xyxy[2]) - int(xyxy[0])) * (int(xyxy[3]) - int(xyxy[1])) >= params.min_head_xyxy:
                                head_helmet[tuple_xyxy(xyxy)] = label
                    else:
                        if names_person[c] in person_list:
                            person[tuple_xyxy(xyxy)] = label
                        if names_person[c] in helmet_head_list:
                            head_helmet[tuple_xyxy(xyxy)] = label
                            pass
                        pass
                    pass
                pass
            pass
        pass

    logger.info(f'Person detect result:： {s}Done.')

    return person, head_helmet
