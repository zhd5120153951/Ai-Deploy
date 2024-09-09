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
        # interval_switch
        if data['tools']['leave_interval_time']['interval_switch'] == 'True':
            self.interval_switch = True
        else:
            self.interval_switch = False
        self.conf_thres_person = float(data['model_args']['conf_thres_person'])
        self.iou_thres_person = float(data['model_args']['iou_thres_person'])
        self.line_thickness = int(data['common_args']['line_thickness'])
        self.min_person_xyxy = float(data['tools']['Target_filter']['min_person_xyxy'])
        self.max_person_xyxy = float(data['tools']['Target_filter']['max_person_xyxy'])
        self.min_head_helmet_xyxy = float(data['tools']['Target_filter']['min_head_helmet_xyxy'])
        self.max_head_helmet_xyxy = float(data['tools']['Target_filter']['max_head_helmet_xyxy'])
        self.people_num = int(data['model_args']['people_num'])
        self.interval_time = float(data['tools']['leave_interval_time']['interval_time'])

    def interval_time(self):
        return self.interval_time

    def interval_switch(self):
        return self.interval_switch

    def people_num(self):
        return self.people_num

    def min_head_helmet_xyxy(self):
        return self.min_head_helmet_xyxy

    def max_head_helmet_xyxy(self):
        return self.max_head_helmet_xyxy
        
    def min_person_xyxy(self):
        return self.min_person_xyxy

    def max_person_xyxy(self):
        return self.max_person_xyxy

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

    def iou_thres_person(self):
        return self.iou_thres_person

    def line_thickness(self):
        return self.line_thickness
