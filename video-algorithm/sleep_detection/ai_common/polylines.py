import cv2


def put_region(img_array_copy, w1, h1, pts, line):
    cv2.putText(img_array_copy, "Detection_Region", (int(img_array_copy.shape[1] * w1 - 5), int(
        img_array_copy.shape[0] * h1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), line, cv2.LINE_AA)
    img_array_copy = cv2.polylines(
        img_array_copy, [pts], True, (0, 255, 0), line)  # 画感兴趣区域
    return img_array_copy
