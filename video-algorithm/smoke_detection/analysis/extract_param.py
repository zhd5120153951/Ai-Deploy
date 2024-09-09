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
        # hand_verify_switch
        if data['tools']['hand_verify']['hand_verify_switch'] == 'True':
            self.hand_verify_switch = True
        else:
            self.hand_verify_switch = False
        # head_verify_switch
        if data['tools']['head_verify']['head_verify_switch'] == 'True':
            self.head_verify_switch = True
        else:
            self.head_verify_switch = False
        self.conf_thres_hand = float(data['model_args']['conf_thres_hand'])
        self.conf_thres_smoke = float(data['model_args']['conf_thres_smoke'])
        self.conf_thres_head = float(data['model_args']['conf_thres_head'])
        self.iou_thres_hand = float(data['model_args']['iou_thres_hand'])
        self.iou_thres_smoke = float(data['model_args']['iou_thres_smoke'])
        self.iou_thres_head = float(data['model_args']['iou_thres_head'])
        self.line_thickness = int(data['common_args']['line_thickness'])
        self.s_h_ratio = float(data['tools']['head_verify']['s_h_ratio'])

    def detect_area_flag(self):
        return self.detect_area_flag

    def hide_labels(self):
        return self.hide_labels

    def hide_conf(self):
        return self.hide_conf

    def polyline(self):
        return self.polyline

    def conf_thres_hand(self):
        return self.conf_thres_hand

    def conf_thres_smoke(self):
        return self.conf_thres_smoke

    def conf_thres_head(self):
        return self.conf_thres_head

    def iou_thres_hand(self):
        return self.iou_thres_hand

    def iou_thres_smoke(self):
        return self.iou_thres_smoke

    def iou_thres_head(self):
        return self.iou_thres_head

    def line_thickness(self):
        return self.line_thickness

    def hand_verify_switch(self):
        return self.hand_verify_switch

    def head_verify_switch(self):
        return self.head_verify_switch

    def s_h_ratio(self):
        return self.s_h_ratio



