from ai_common import setting
from ai_common.img_util import similarity, preprocess_np, preprocess_tensor
from ai_common.log_util import logger
from ai_common.util.general import non_max_suppression
from ai_common.util.torch_utils import time_synchronized


def img_preprocess(device, data, conf_thres, iou_thres, MODEL, engine):
    pred = []
    degree = 0
    ts = time_synchronized()
    img = data['img']
    img_array = data['img_array']
    half = False

    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # names = MODEL.module.names if hasattr(MODEL,'module') else MODEL.names  # ['person', 'head', 'helmet']
    names = ['person', 'head', 'helmet']

    if half:
        MODEL.half()

    if data['tools']['keyframe']['key_switch']:
        if data['k8sName'] in setting.last_hist_dic.keys():
            degree, hist = similarity(img, setting.last_hist_dic[data['k8sName']])
            if degree != 0 and (degree > float(data['tools']['keyframe']['degree^']) or degree < float(
                    data['tools']['keyframe']['degree_'])):  # 相似度
                t3 = time_synchronized()
                logger.debug(f'设备 {data["k8sName"]} 相似帧过滤耗时： ({t3 - ts:.3f})s.')
                return pred, names, img, img_array, degree
            else:
                setting.last_hist_dic[data['k8sName']] = hist

                if data['common_args']['tensorrt'] == True:
                    from analysis.trt_inference import trt_infer
                    img = preprocess_np(img)
                    # Tensorrt Inference
                    pred = trt_infer(data, engine, img)
                else:
                    img = preprocess_tensor(img, device, half)
                    # Pytorch Inference
                    t4 = time_synchronized()
                    pred = MODEL(img, augment=False)[0]
                    t5 = time_synchronized()
                    logger.info(f'设备 {data["k8sName"]} Pytorch推理预测耗时： ({t5 - t4:.3f})s.')

                # Apply NMS
                pred = non_max_suppression(pred, conf_thres, iou_thres)
            pass
        pass
    else:
        if data['common_args']['tensorrt'] == True:
            from analysis.trt_inference import trt_infer
            img = preprocess_np(img)
            # Tensorrt Inference
            pred = trt_infer(data, engine, img)
        else:
            img = preprocess_tensor(img, device, half)
            # Pytorch Inference
            t4 = time_synchronized()
            pred = MODEL(img, augment=False)[0]
            t5 = time_synchronized()
            logger.info(f'设备 {data["k8sName"]} Pytorch推理预测耗时： ({t5 - t4:.3f})s.')
            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres)
            pass
        pass
    t8 = time_synchronized()
    logger.info(f'设备 {data["k8sName"]} 模型处理图片总耗时 ({t8 - ts:.3f})s.')
    return pred, names, img, img_array, degree