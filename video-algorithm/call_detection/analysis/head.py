from ai_common.log_util import logger
from ai_common.np_util import person_in_poly_area, is_poi_in_poly, tuple_xyxy, load_poly_area_data
from ai_common.util.general import scale_coords


def process_result_head(pred_gathered, detect_area_flag, names_gathered, hide_labels, hide_conf, data, img, img_array):
    img_array_copy = img_array.copy()
    xyxy_lab_dic = {}
    head = {}
    sg = ''
    pts, w1, h1 = load_poly_area_data(data)
    for i, det in enumerate(pred_gathered):
        sg += '%g x %g ' % img.shape[2:]  # logger.info string, 取第三和第四维，对应宽和高 512 x 640

        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_array_copy.shape).round()

            # logger.info results
            for cg in det[:,-1].unique():
                ng = (det[:, -1] == cg).sum()
                sg += f"{ng} {names_gathered[int(cg)]}{'s' * (ng > 1)}, "

            # judge results
            for *xyxy, conf, cls in reversed(det):
                head_list = ['head']
                cg = int(cls)  # integer class 1， 0
                xyxy = [xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()]

                # label
                if hide_labels:
                    label = None
                elif hide_conf:
                    label = names_gathered[cg]
                else:
                    label = names_gathered[cg] + "%.2f" % conf

                if detect_area_flag:
                    # 求物体框的中心点
                    object_cx, object_cy = person_in_poly_area(xyxy)
                    # 判断中心点是否在检测框内部
                    if not is_poi_in_poly([object_cx, object_cy], pts):
                        # 不在感兴趣的框内，则继续判断下一个物体。
                        continue
                    # 用于抽烟检测传参
                    if names_gathered[cg] in head_list:
                        if data['tools']['Target_filter']['filter_switch'] == True:
                            if int(data['tools']['Target_filter']['max_head_xyxy']) > (int(xyxy[2]) - int(xyxy[0])) * (int(xyxy[3]) - int(xyxy[1])) > int(data['tools']['Target_filter']['min_head_xyxy']):
                                head[tuple_xyxy(xyxy)] = label
                                pass
                            pass
                        else:
                            head[tuple_xyxy(xyxy)] = label
                            pass
                        pass
                    pass
                else:
                    if names_gathered[cg] in head_list:
                        if data['tools']['Target_filter']['filter_switch'] == True:
                            if int(data['tools']['Target_filter']['max_head_xyxy']) > (int(xyxy[2]) - int(xyxy[0])) * (int(xyxy[3]) - int(xyxy[1])) > int(data['tools']['Target_filter']['min_head_xyxy']):
                                head[tuple_xyxy(xyxy)] = label
                                pass
                            pass
                        else:
                            head[tuple_xyxy(xyxy)] = label
                            pass
                        pass
                    pass
                pass
            pass
        pass
    logger.info(f'Gathered detect result: {sg}Done.')

    return head, img_array_copy
