import random

from ai_common import setting
from ai_common.img_util import similarity, preprocess_tensor, preprocess_np
from ai_common.log_util import logger
from ai_common.util.general import non_max_suppression
from ai_common.util.torch_utils import time_synchronized


def img_preprocess(device, data, conf_thres_car, iou_thres_car, MODEL_det, engine_det, engine_rec):
    pred_car = []
    degree = 0
    ts = time_synchronized()
    img = data['img']
    img_array = data['img_array']
    half = False
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # names_car = MODEL_det.module.names if hasattr(MODEL_det, 'module') else MODEL_det.names
    names_car = ['plate']
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names_car))]

    if half:
        MODEL_det.half()  # to FP16


    if data['tools']['keyframe']['key_switch'] == True:
        if data['k8sName'] in setting.last_hist_dic.keys():
            degree, hist = similarity(img, setting.last_hist_dic[data['k8sName']])
            if degree != 0 and (degree > float(data['tools']['keyframe']['degree^']) or degree < float(
                    data['tools']['keyframe']['degree_'])):  # 相似度
                t3 = time_synchronized()
                logger.debug(f'设备 {data["k8sName"]} 相似帧过滤耗时： ({t3 - ts:.3f})s.')
                return pred_car, names_car, img, img_array, degree
            else:
                setting.last_hist_dic[data['k8sName']] = hist

                if data['common_args']['tensorrt'] == True:
                    from analysis.trt_inference import trt_infer
                    img = preprocess_np(img)
                    # Tensorrt Inference
                    pred_car = trt_infer(data, engine_det, img)
                else:
                    img = preprocess_tensor(img, device, half)
                    # Pytorch Inference
                    t4 = time_synchronized()
                    pred_car = MODEL_det(img, augment=False)[0]
                    t5 = time_synchronized()
                    logger.info(f'设备 {data["k8sName"]} Pytorch推理预测耗时： ({t5 - t4:.3f})s.')

                # Apply NMS
                t6 = time_synchronized()
                pred_car = non_max_suppression(pred_car, conf_thres_car, iou_thres_car)
                t7 = time_synchronized()
                logger.debug(f'设备 {data["k8sName"]} 推理结果NMS耗时 ({t7 - t6:.3f})s.')
                pass
            pass
        pass
    else:
        if data['common_args']['tensorrt'] == True:
            from analysis.trt_inference import trt_infer
            img = preprocess_np(img)
            # Tensorrt Inference
            pred_car = trt_infer(data, engine_det, img)
        else:
            img = preprocess_tensor(img, device, half)
            # Pytorch Inference
            t4 = time_synchronized()
            pred_car = MODEL_det(img, augment=False)[0]
            t5 = time_synchronized()
            logger.info(f'设备 {data["k8sName"]} Pytorch推理预测耗时： ({t5 - t4:.3f})s.')

        # Apply NMS
        t6 = time_synchronized()
        pred_car = non_max_suppression(pred_car, conf_thres_car, iou_thres_car)
        t7 = time_synchronized()
        logger.debug(f'设备 {data["k8sName"]} 推理结果NMS耗时 ({t7 - t6:.3f})s.')

    t8 = time_synchronized()
    logger.info(f'设备 {data["k8sName"]} 模型处理图片总耗时 ({t8 - ts:.3f})s.')
    return pred_car, names_car, img, img_array, degree, colors