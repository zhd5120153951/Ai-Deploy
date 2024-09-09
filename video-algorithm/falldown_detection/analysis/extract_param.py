def param(data):
    # detect_area_flag
    if data['tools']['detect_area_flag']['area_switch'] == 'True':
        detect_area_flag = True
    else:
        detect_area_flag = False
    # polyline
    if data['common_args']['polyline'] == 'True':
        polyline = True
    else:
        polyline = False
    standbbx_ratio = float(data['common_args']['standbbx_ratio'])
    clear_t = float(data['common_args']['clear_t'])
    h_h_conf = float(data['common_args']['h_h_conf'])
    cal_ious = float(data['common_args']['cal_ious'])
    d_frame = float(data['common_args']['d_frame'])
    conf_thres = float(data['model_args']['conf_thres'])
    iou_thres = float(data['model_args']['iou_thres'])
    line_thickness = int(data['common_args']['line_thickness'])

    return detect_area_flag, polyline, standbbx_ratio, clear_t, h_h_conf, cal_ious, d_frame, conf_thres, iou_thres, line_thickness