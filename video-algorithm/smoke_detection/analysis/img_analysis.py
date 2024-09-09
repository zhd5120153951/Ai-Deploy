from ai_common import setting
from ai_common.img_util import similarity, preprocess_np, preprocess_tensor
from ai_common.log_util import logger
from ai_common.util.general import non_max_suppression
from ai_common.util.torch_utils import time_synchronized
from analysis.extract_param import param


def img_preprocess(device, data, MODEL_hand, MODEL_smoke, MODEL_head, engine_hand, engine_smoke, engine_head):
    params = param(data)

    pred_hand = []
    pred_smoke = []
    pred_head = []
    degree = 0
    ts = time_synchronized()
    img = data['img']
    img_array = data['img_array']
    half = False
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    
    # names_hand = MODEL_hand.module.names if hasattr(MODEL_hand, 'module') else MODEL_hand.names
    # names_smoke = MODEL_smoke.module.names if hasattr(MODEL_smoke, 'module') else MODEL_smoke.names  # ['smoke']
    # names_head = MODEL_head.module.names if hasattr(MODEL_head, 'module') else MODEL_head.names
    names_hand = ['hand']
    names_smoke = ['smoke']
    names_head = ['person', 'head', 'helmet']

    if half:
        MODEL_hand.half()
        MODEL_smoke.half()
        MODEL_head.half()

    if data['tools']['keyframe']['key_switch']:
        if data['k8sName'] in setting.last_hist_dic.keys():
            degree, hist = similarity(img, setting.last_hist_dic[data['k8sName']])
            if degree != 0 and (degree > float(data['tools']['keyframe']['degree^']) or degree < float(
                    data['tools']['keyframe']['degree_'])):  # 相似度
                t3 = time_synchronized()
                logger.debug(f'设备 {data["k8sName"]} 相似帧过滤耗时： ({t3 - ts:.3f})s.')
                return pred_hand, pred_smoke, pred_head, names_hand, names_smoke, names_head, img, img_array, degree
            else:
                setting.last_hist_dic[data['k8sName']] = hist

                if data['common_args']['tensorrt'] == True:
                    from analysis.trt_inference import trt_infer
                    img = preprocess_np(img)
                    # Tensorrt Inference
                    pred_hand = trt_infer(data, engine_hand, img)
                    pred_smoke = trt_infer(data, engine_smoke, img)
                    pred_head = trt_infer(data, engine_head, img)
                else:
                    img = preprocess_tensor(img, device, half)
                    # Pytorch Inference
                    t4 = time_synchronized()
                    pred_hand = MODEL_hand(img, augment=False)[0]
                    pred_smoke = MODEL_smoke(img, augment=False)[0]
                    pred_head = MODEL_head(img, augment=False)[0]
                    t5 = time_synchronized()
                    logger.info(f'设备 {data["k8sName"]} Pytorch推理预测耗时： ({t5 - t4:.3f})s.')

                # Apply NMS
                pred_hand = non_max_suppression(pred_hand, params.conf_thres_hand, params.iou_thres_hand)
                pred_smoke = non_max_suppression(pred_smoke, params.conf_thres_smoke, params.iou_thres_smoke)
                pred_head = non_max_suppression(pred_head, params.conf_thres_head, params.iou_thres_head)
            pass
        pass
    else:
        if data['common_args']['tensorrt'] == True:
            from analysis.trt_inference import trt_infer
            img = preprocess_np(img)
            # Tensorrt Inference
            pred_hand = trt_infer(data, engine_hand, img)
            pred_smoke = trt_infer(data, engine_smoke, img)
            pred_head = trt_infer(data, engine_head, img)
        else:
            img = preprocess_tensor(img, device, half)
            # Pytorch Inference
            t4 = time_synchronized()
            pred_hand = MODEL_hand(img, augment=False)[0]
            pred_smoke = MODEL_smoke(img, augment=False)[0]
            pred_head = MODEL_head(img, augment=False)[0]
            t5 = time_synchronized()
            logger.info(f'设备 {data["k8sName"]} Pytorch推理预测耗时： ({t5 - t4:.3f})s.')

            # Apply NMS
            t6 = time_synchronized()
            pred_hand = non_max_suppression(pred_hand, params.conf_thres_hand, params.iou_thres_hand)
            pred_smoke = non_max_suppression(pred_smoke, params.conf_thres_smoke, params.iou_thres_smoke)
            pred_head = non_max_suppression(pred_head, params.conf_thres_head, params.iou_thres_head)
            t7 = time_synchronized()
            logger.info(f'设备 {data["k8sName"]} 推理结果NMS耗时 ({t7 - t6:.3f})s.')
            pass
        pass
    t8 = time_synchronized()
    logger.info(f'设备 {data["k8sName"]} 模型处理图片总耗时 ({t8 - ts:.3f})s.')
    return pred_hand, pred_smoke, pred_head, names_hand, names_smoke, names_head, img, img_array, degree