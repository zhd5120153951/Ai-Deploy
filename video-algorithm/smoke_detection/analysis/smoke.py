from ai_common.log_util import logger
from ai_common.np_util import person_in_poly_area, is_poi_in_poly, tuple_xyxy, cal_iou, load_poly_area_data
from ai_common.util.general import scale_coords
from analysis.extract_param import param


def process_result_smoke(pred_smoke, names_smoke, data, img, img_array):
    params = param(data)
    smoke = {}
    img_array_copy = img_array.copy()
    s = ''
    pts, w1, h1 = load_poly_area_data(data)
    for i, det in enumerate(pred_smoke):
        img_array_copy = img_array.copy()
        s += '%g x %g ' % img.shape[2:]  # print string, 取第三和第四维，对应宽和高 512 x 640
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_array_copy.shape).round()

        # logger.info results
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()
            s += f"{n} {names_smoke[int(c)]}{'s' * (n > 1)}, "

        # Write results
        for *xyxy, conf, cls in det:
            c = int(cls)
            xyxy = [xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()]

            # label
            if params.hide_labels:
                label = None
            elif params.hide_conf:
                label = names_smoke[c]
            else:
                label = names_smoke[c] + "%.2f" % conf

            smoke_list = ['smoke']
            if params.detect_area_flag:
                # 求物体框的中心点
                object_cx, object_cy = person_in_poly_area(xyxy)
                # 判断中心点是否在检测框内部
                if not is_poi_in_poly([object_cx, object_cy], pts):
                    # 不在感兴趣的框内，则继续判断下一个物体。
                    continue
                if names_smoke[c] in smoke_list:
                    if data['tools']['Target_filter']['filter_switch'] == True:
                        if (int(xyxy[2]) - int(xyxy[0])) * (int(xyxy[3]) - int(xyxy[1])) > int(data['min_people_xyxy']):
                            smoke[tuple_xyxy(xyxy)] = label
                    else:
                        smoke[tuple_xyxy(xyxy)] = label
            else:
                if names_smoke[c] in smoke_list:
                    if data['tools']['Target_filter']['filter_switch'] == True:
                        if (int(xyxy[2]) - int(xyxy[0])) * (int(xyxy[3]) - int(xyxy[1])) > int(data['min_people_xyxy']):
                            smoke[tuple_xyxy(xyxy)] = label
                    else:
                        smoke[tuple_xyxy(xyxy)] = label

    logger.info(f'Smoke detect result: {s}Done.')

    # iou_list = []
    # # 对头和香烟进行判断
    # for h in head:
    #     for s in smoke:
    #         iou = cal_iou(h, s)
    #         if iou > 0:
    #             iou_list.append(iou)
    #             s_h = (h[2] - h[0]) * (h[3] - h[1])
    #             s_s = (s[2] - s[0]) * (s[3] - s[1])
    #             h_divide_s = s_s / s_h
    #             if h_divide_s < float(data['s/h']):
    #                 flag_smoke = False
    #                 plot_one_box(h, img_array_copy, label=head[h], color=(255, 0, 0), line_thickness=params.line_thickness)
    #                 plot_one_box(s, img_array_copy, label=smoke[s], color=(0, 0, 255), line_thickness=params.line_thickness)
    #
    # logger.info(f'Smoke detect result: {s}Done.')
    #
    # if setting.VERIFY == 1:
    #     if params.detect_area_flag:
    #         if params.polyline:
    #             img_array_copy = put_region(img_array_copy, w1, h1, pts, params.line_thickness)
    #     cv2.imshow('smoke_detect', img_array_copy)
    #     cv2.waitKey()
    #     cv2.destroyAllWindows()

    # if not flag_smoke:
    #     if params.detect_area_flag:
    #         if params.polyline:
    #             img_array_copy = put_region(img_array_copy, w1, h1, pts, params.line_thickness)
    #     logger.info('Detect stuff smoking!')
    #     alarm_label = 'smoke'
    #     alarm_type_smoking = "抽烟"
    #     alarm_info_smoking = "{\"AlarmMsg\": \"Detect stuff smoking.\"}"
    #     # 组装告警信息
    #     package_business_alarm(alarm_info_smoking, alarm_type_smoking, img_array_copy, img_array, data['deviceId'], alarm_label)

    return smoke, img_array_copy