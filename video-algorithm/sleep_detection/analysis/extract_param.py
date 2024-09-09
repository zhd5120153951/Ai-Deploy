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
        self.iou_thres_person = float(data['model_args']['iou_thres_person'])
        self.line_thickness = int(data['common_args']['line_thickness'])
        self.cal_ious = float(data['model_args']['cal_ious'])
        self.sames = float(data['model_args']['sames'])
        self.sleep_interval_time = float(data['model_args']['sleep_interval_time'])

    def sames(self):
        return self.sames

    def cal_ious(self):
        return self.cal_ious

    def sleep_interval_time(self):
        return self.sleep_interval_time

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
