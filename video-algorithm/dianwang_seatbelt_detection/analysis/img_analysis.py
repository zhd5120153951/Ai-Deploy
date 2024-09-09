import random

from ai_common import setting
from ai_common.img_util import similarity, preprocess_np, preprocess_tensor
from ai_common.log_util import logger
from ai_common.util.general import non_max_suppression
from ai_common.util.torch_utils import time_synchronized
from yolov3_utils.utils import preprocess_tensor_v3


def img_preprocess(device, data, params, MODEL_person, MODEL_seatbelt, MODEL_high, engine_person, engine_seatbelt, engine_high):
    pred_person = []
    pred_seatbelt = []
    pred_high = []
    degree = 0
    ts = time_synchronized()
    img = data['img']
    img_array = data['img_array']
    half = False
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # names_ = MODEL_seatbelt.module.names if hasattr(MODEL_seatbelt, 'module') else MODEL_seatbelt.names # ['badge', 'offground', 'ground', 'safebelt']
    names = ["nosafebelt","safebelt"]
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    if half:
        MODEL_person.half()
        MODEL_seatbelt.half()
        MODEL_high.half()

    if data['tools']['keyframe']['key_switch']:
        if data['k8sName'] in setting.last_hist_dic.keys():
            degree, hist = similarity(img, setting.last_hist_dic[data['k8sName']])
            if degree != 0 and (degree > float(data['tools']['keyframe']['degree^']) or degree < float(
                    data['tools']['keyframe']['degree_'])):  # 相似度
                t3 = time_synchronized()
                logger.debug(f'设备 {data["k8sName"]} 相似帧过滤耗时： ({t3 - ts:.3f})s.')
                return pred_person, pred_seatbelt, pred_high, names, img, img_array, degree, colors
            else:
                setting.last_hist_dic[data['k8sName']] = hist

                if data['common_args']['tensorrt'] == True:
                    from analysis.trt_inference import trt_infer
                    img = preprocess_np(img)
                    # Tensorrt Inference
                    pred_person = trt_infer(data, engine_person, img)
                    pred_seatbelt = trt_infer(data, engine_seatbelt, img)
                    pred_high = trt_infer(data, engine_high, img)
                else:
                    img = preprocess_tensor(img, device, half)
                    img0 = preprocess_tensor_v3(img_array)
                    # Pytorch Inference
                    t4 = time_synchronized()
                    pred_person = MODEL_person(img, augment=False)[0]
                    pred_seatbelt = MODEL_seatbelt(img, augment=False)[0]
                    pred_high = MODEL_high(img0)
                    t5 = time_synchronized()
                    logger.info(f'设备 {data["k8sName"]} Pytorch推理预测耗时： ({t5 - t4:.3f})s.')

                # Apply NMS
                pred_person = non_max_suppression(pred_person, params.conf_thres_person, params.iou_thres_person)
                pred_seatbelt = non_max_suppression(pred_seatbelt, params.conf_thres_seatbelt, params.iou_thres_seatbelt)
                pred_high = non_max_suppression(pred_high, params.conf_thres_high, params.iou_thres_high)
            pass
        pass
    else:
        if data['common_args']['tensorrt'] == True:
            from analysis.trt_inference import trt_infer
            img = preprocess_np(img)
            # Tensorrt Inference
            pred_person = trt_infer(data, engine_person, img)
            pred_seatbelt = trt_infer(data, engine_seatbelt, img)
            pred_high = trt_infer(data, engine_high, img)
        else:
            img = preprocess_tensor(img, device, half)
            # Pytorch Inference
            t4 = time_synchronized()
            pred_person = MODEL_person(img, augment=False)[0]
            pred_seatbelt = MODEL_seatbelt(img, augment=False)[0]
            pred_high = MODEL_high(img)
            t5 = time_synchronized()
            logger.info(f'设备 {data["k8sName"]} Pytorch推理预测耗时： ({t5 - t4:.3f})s.')

            # Apply NMS
            t6 = time_synchronized()
            pred_person = non_max_suppression(pred_person, params.conf_thres_person, params.iou_thres_person)
            pred_seatbelt = non_max_suppression(pred_seatbelt, params.conf_thres_seatbelt, params.iou_thres_seatbelt)
            pred_high = non_max_suppression(pred_high, params.conf_thres_high, params.iou_thres_high)
            t7 = time_synchronized()
            logger.info(f'设备 {data["k8sName"]} 推理结果NMS耗时 ({t7 - t6:.3f})s.')
            pass
        pass
    t8 = time_synchronized()
    logger.info(f'设备 {data["k8sName"]} 模型处理图片总耗时 ({t8 - ts:.3f})s.')
    return pred_person, pred_seatbelt, pred_high, names, img, img0,img_array, degree, colors