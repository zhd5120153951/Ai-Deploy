from ai_common import setting
from ai_common.img_util import similarity, preprocess_tensor, preprocess_np
from ai_common.log_util import logger
from ai_common.util.general import non_max_suppression
from ai_common.util.torch_utils import time_synchronized


def img_preprocess(device, data, conf_thres_head, conf_thres_call, iou_thres_head, iou_thres_call, MODEL_head, MODEL_call, engine_call, engine_head):
    pred_call = []
    pred_head = []
    degree = 0
    ts = time_synchronized()
    img = data['img']
    img_array = data['img_array']
    half = False

    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # names_head = MODEL_head.module.names if hasattr(MODEL_head, 'module') else MODEL_head.names
    # names_call = MODEL_call.module.names if hasattr(MODEL_call, 'module') else MODEL_call.names
    names_head = ['person', 'head', 'helmet']
    names_call = ['phone']

    if half:
        MODEL_call.half()
        MODEL_head.half()

    if data['tools']['keyframe']['key_switch']:
        if data['k8sName'] in setting.last_hist_dic.keys():
            degree, hist = similarity(img, setting.last_hist_dic[data['k8sName']])
            if degree != 0 and (degree > float(data['tools']['keyframe']['degree^']) or degree < float(data['tools']['keyframe']['degree_'])):  # 相似度
                t3 = time_synchronized()
                logger.debug(f'设备 {data["k8sName"]} 相似帧过滤耗时： ({t3 - ts:.3f})s.')
                return pred_call, pred_head, names_call, names_head, img, img_array, degree
            else:
                setting.last_hist_dic[data['k8sName']] = hist

                if data['common_args']['tensorrt'] == True:
                    from analysis.trt_inference import trt_infer
                    img = preprocess_np(img)
                    # Tensorrt Inference
                    pred_head = trt_infer(data, engine_head, img)
                    pred_call = trt_infer(data, engine_call, img)
                else:
                    img = preprocess_tensor(img, device, half)
                    # Pytorch Inference
                    t4 = time_synchronized()
                    pred_head = MODEL_head(img, augment=False)[0]
                    pred_call = MODEL_call(img, augment=False)[0]
                    t5 = time_synchronized()
                    logger.info(f'设备 {data["k8sName"]} Pytorch推理预测耗时： ({t5 - t4:.3f})s.')

                # Apply NMS
                pred_head = non_max_suppression(pred_head, conf_thres_head, iou_thres_head)
                pred_call = non_max_suppression(pred_call, conf_thres_call, iou_thres_call)
            pass
        pass
    else:
        if data['common_args']['tensorrt'] == True:
            from analysis.trt_inference import trt_infer
            img = preprocess_np(img)
            # Tensorrt Inference
            pred_head = trt_infer(data, engine_head, img)
            pred_call = trt_infer(data, engine_call, img)
        else:
            img = preprocess_tensor(img, device, half)
            # Pytorch Inference
            t4 = time_synchronized()
            pred_head = MODEL_head(img, augment=False)[0]
            pred_call = MODEL_call(img, augment=False)[0]
            t5 = time_synchronized()
            logger.info(f'设备 {data["k8sName"]} Pytorch推理预测耗时： ({t5 - t4:.3f})s.')

            # Apply NMS
        pred_head = non_max_suppression(pred_head, conf_thres_head, iou_thres_head)
        pred_call = non_max_suppression(pred_call, conf_thres_call, iou_thres_call)

    t8 = time_synchronized()
    logger.info(f'设备 {data["k8sName"]} 模型处理图片总耗时 ({t8 - ts:.3f})s.')
    return pred_call, pred_head, names_call, names_head, img, img_array, degree