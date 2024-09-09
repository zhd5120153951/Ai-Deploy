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
            
        self.conf_thres_person = float(data['model_args']['conf_thres_person'])
        self.conf_thres_uniform = float(data['model_args']['conf_thres_uniform'])
        self.iou_thres_person = float(data['model_args']['iou_thres_person'])
        self.iou_thres_uniform = float(data['model_args']['iou_thres_uniform'])
        self.line_thickness = int(data['common_args']['line_thickness'])
        self.min_person_xyxy = float(data['tools']['Target_filter']['min_person_xyxy'])
        self.max_person_xyxy = float(data['tools']['Target_filter']['max_person_xyxy'])
        self.min_head_xyxy = float(data['tools']['Target_filter']['min_head_xyxy'])
        self.max_person_xyxy = float(data['tools']['Target_filter']['max_person_xyxy'])
        self.p_h_iou_min = float(data['tools']['judge']['p_h_iou_min'])
        self.p_h_iou_max = float(data['tools']['judge']['p_h_iou_max'])
        self.p_u_iou = float(data['tools']['judge']['p_u_iou'])

    def p_h_iou_min(self):
        return self.p_h_iou_min

    def p_h_iou_max(self):
        return self.p_h_iou_max

    def p_u_iou(self):
        return self.p_u_iou

    def detect_area_flag(self):
        return self.detect_area_flag

    def hide_labels(self):
        return self.hide_labels

    def hide_conf(self):
        return self.hide_conf

    def polyline(self):
        return self.polyline

    def conf_thres_person(self):
        return self.conf_thres_person

    def conf_thres_uniform(self):
        return self.conf_thres_uniform

    def iou_thres_person(self):
        return self.iou_thres_person

    def iou_thres_uniform(self):
        return self.iou_thres_uniform

    def line_thickness(self):
        return self.line_thickness

    def person_verify_switch(self):
        return self.person_verify_switch

    def min_person_xyxy(self):
        return self.min_person_xyxy

    def max_person_xyxy(self):
        return self.max_person_xyxy

    def min_head_xyxy(self):
        return self.min_head_xyxy

    def max_head_xyxy(self):
        return self.max_head_xyxy


