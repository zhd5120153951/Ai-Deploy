class param(object):
    def __init__(self, data):
        # detect_area_flag
        if data['tools']['detect_area_flag']['area_switch'] == 'True':
            self.detect_area_flag = True
        else:
            self.detect_area_flag = False
        # hide_labels
        if data['common_args']['hide_labels'] == 'True':
            self.hide_labels = True
        else:
            self.hide_labels = False
        # hide_conf
        if data['common_args']['hide_conf'] == 'True':
            self.hide_conf = True
        else:
            self.hide_conf = False
        # polyline
        if data['common_args']['polyline'] == 'True':
            self.polyline = True
        else:
            self.polyline = False
        self.conf_thres_mask = float(data['model_args']['conf_thres_mask'])
        self.iou_thres_mask = float(data['model_args']['iou_thres_mask'])
        self.line_thickness = int(data['common_args']['line_thickness'])
        self.min_mask_xyxy = float(data['tools']['Target_filter']['min_mask_xyxy'])
        self.max_mask_xyxy = float(data['tools']['Target_filter']['max_mask_xyxy'])
        
    def min_mask_xyxy(self):
        return self.min_mask_xyxy

    def max_mask_xyxy(self):
        return self.max_mask_xyxy

    def detect_area_flag(self):
        return self.detect_area_flag

    def hide_labels(self):
        return self.hide_labels

    def hide_conf(self):
        return self.hide_conf

    def polyline(self):
        return self.polyline

    def conf_thres_mask(self):
        return self.conf_thres_mask

    def iou_thres_mask(self):
        return self.iou_thres_mask

    def line_thickness(self):
        return self.line_thickness
