import cv2


class MOG2Detector:
    def __init__(self, history, dist2Threshold, minArea):
        self.minArea = minArea
        self.dector = cv2.createBackgroundSubtractorKNN(history, dist2Threshold, False) # 生成前景掩码 / 参数解释（历史帧数，阈值，阴影检测）

    def detectOneFrame(self, frame):
        if frame is None:
            return None
        mask = self.dector.apply(frame) # 计算前景掩码和更新背景

        # 平滑处理
        element = cv2.blur(frame, (3, 3))  # 均值滤波
        element = cv2.boxFilter(element, -1, (2, 2), normalize=False)  # 方框滤波

        # 形态学去噪
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 得到一个卷积核，用于后续的腐蚀，膨胀，开，闭运算 -> 返回卷积核的大小
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, element) # 开运算，先腐蚀后膨胀。去除黑点
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, element) # 膨胀

        # 寻找轮廓
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxs = []
        for contour in contours:
            if cv2.contourArea(contour) >= self.minArea: # 轮廓面积
                bboxs.append(cv2.boundingRect(contour))
        return mask, bboxs
