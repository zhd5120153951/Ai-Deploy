def param(data):
    # detect_area_flag
    if data['tools']['detect_area_flag']['area_switch'] == 'True':
        detect_area_flag = True
    else:
        detect_area_flag = False
    # hide_labels
    if data['common_args']['hide_labels'] == 'True':
        hide_labels = True
    else:
        hide_labels = False
    # hide_conf
    if data['common_args']['hide_conf'] == 'True':
        hide_conf = True
    else:
        hide_conf = False
    # polyline
    if data['common_args']['polyline'] == 'True':
        polyline = True
    else:
        polyline = False
    conf_thres_gathered = float(data['model_args']['conf_thres_gathered'])
    iou_thres_gathered = float(data['model_args']['iou_thres_gathered'])
    line_thickness = int(data['common_args']['line_thickness'])

    return detect_area_flag, hide_labels, hide_conf, polyline, conf_thres_gathered, iou_thres_gathered, line_thickness