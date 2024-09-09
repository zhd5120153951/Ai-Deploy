from ai_common.np_util import person_in_poly_area, is_poi_in_poly, tuple_xyxy, load_poly_area_data
from analysis.extract_param import param
from ai_common.util.general import scale_coords


def process_result_high(pred_high, data, img, img_array):
    params = param(data)
    img_array_copy = img_array.copy()
    high = {}
    pts, w1, h1 = load_poly_area_data(data)
    for i, det in enumerate(pred_high):
        if len(det):
            # det = det.numpy()
            # det[:, :4] = rescale_boxes(det, 416, img_array_copy.shape[:2])
            # det = torch.from_numpy(det)
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_array_copy.shape).round()

            # judge results
            for *xyxy, conf, cls in reversed(det):
                c = int(cls) # high -> cls == 0
                xyxy = [xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()]

                label = str(c) + "%.2f" % conf

                if params.detect_area_flag:
                    # 求物体框的中心点
                    object_cx, object_cy = person_in_poly_area(xyxy)
                    # 判断中心点是否在检测框内部
                    if not is_poi_in_poly([object_cx, object_cy], pts):
                        # 不在感兴趣的框内，则继续判断下一个物体。
                        continue
                    # 传参
                    if c == 0:
                        if data['tools']['Target_filter']['filter_switch'] == True:
                            if int(data['tools']['Target_filter']['max_high_xyxy']) > (int(xyxy[2]) - int(xyxy[0])) * (int(xyxy[3]) - int(xyxy[1])) > int(data['tools']['Target_filter']['min_high_xyxy']):
                                high[tuple_xyxy(xyxy)] = label
                                pass
                            pass
                        else:
                            high[tuple_xyxy(xyxy)] = label
                            pass
                        pass
                    pass
                else:
                    if c == 0:
                        if data['tools']['Target_filter']['filter_switch'] == True:
                            if int(data['tools']['Target_filter']['max_high_xyxy']) > (int(xyxy[2]) - int(xyxy[0])) * (int(xyxy[3]) - int(xyxy[1])) > int(data['tools']['Target_filter']['min_high_xyxy']):
                                high[tuple_xyxy(xyxy)] = label
                                pass
                            pass
                        else:
                            high[tuple_xyxy(xyxy)] = label
                            pass
                        pass
                    pass
                pass
            pass
        pass

    return high, img_array_copy
