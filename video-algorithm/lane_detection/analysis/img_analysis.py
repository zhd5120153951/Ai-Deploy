import cv2

from ai_common import setting


def similarity(img, nparray):
    im0s = img.copy()
    im0s = cv2.cvtColor(im0s, cv2.COLOR_RGB2BGR)
    if len(nparray) == 0:
        H1 = cv2.calcHist([im0s], [1], None, [256], [0, 256])
        degree = 0
        return degree, H1
    else:
        degree = 0
        H1 = cv2.calcHist([im0s], [1], None, [256], [0, 256])
        for i in range(len(H1)):
            if H1[i] != nparray[i]:
                degree = degree + (1 - abs(H1[i] - nparray[i]) / max(H1[i], nparray[i]))
            else:
                degree += 1
        degree = degree / len(H1)
        return degree, H1

def img_preprocess(data):
    img_array = data['img_array']
    degree = 0
    flag = True

    if data['tools']['keyframe']['key_switch'] == True:
        if data['k8sName'] in setting.last_hist_dic.keys():
            degree, hist = similarity(img_array, setting.last_hist_dic[data['k8sName']])
            if degree != 0 and (degree > float(data['tools']['keyframe']['degree^']) or degree < float(data['tools']['keyframe']['degree_'])):  # 相似度
                flag = False
                return img_array, degree, flag
            else:
                setting.last_hist_dic[data['k8sName']] = hist
                flag = True
    else:
        pass

    return img_array, degree, flag

