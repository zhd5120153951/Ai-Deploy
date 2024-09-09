from ai_common import setting
from ai_common.img_util import similarity, preprocess_np, preprocess_tensor
from ai_common.log_util import logger
from ai_common.util.general import non_max_suppression
from ai_common.util.torch_utils import time_synchronized


def img_preprocess(device, data, MODEL_person, MODEL_uniform, engine_person, engine_uniform, params):
    pred_person = []
    pred_uniform = []
    degree = 0
    ts = time_synchronized()
    img = data['img']
    img_array = data['img_array']
    half = False
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # names_person = MODEL_person.module.names if hasattr(MODEL_person, 'module') else MODEL_person.names
    # names_uniform = MODEL_uniform.module.names if hasattr(MODEL_uniform, 'module') else MODEL_uniform.names
    names_person = ['person', 'head', 'helmet']
    names_uniform = ['uniform', 'nouniform']
    if half:
        MODEL_person.half()
        MODEL_uniform.half()

    if data['tools']['keyframe']['key_switch']:
        if data['k8sName'] in setting.last_hist_dic.keys():
            degree, hist = similarity(img, setting.last_hist_dic[data['k8sName']])
            if degree != 0 and (degree > float(data['tools']['keyframe']['degree^']) or degree < float(
                    data['tools']['keyframe']['degree_'])):  # 相似度
                t3 = time_synchronized()
                logger.debug(f'设备 {data["k8sName"]} 相似帧过滤耗时： ({t3 - ts:.3f})s.')
                return pred_person, pred_uniform, names_person, names_uniform, img, img_array, degree
            else:
                setting.last_hist_dic[data['k8sName']] = hist

                if data['common_args']['tensorrt'] == True:
                    from analysis.trt_inference import trt_infer
                    img = preprocess_np(img)
                    # Tensorrt Inference
                    pred_person = trt_infer(data, engine_person, img)
                    pred_uniform = trt_infer(data, engine_uniform, img)
                else:
                    img = preprocess_tensor(img, device, half)
                    # Pytorch Inference
                    t4 = time_synchronized()
                    pred_person = MODEL_person(img, augment=False)[0]
                    pred_uniform = MODEL_uniform(img, augment=False)[0]
                    t5 = time_synchronized()
                    logger.info(f'设备 {data["k8sName"]} Pytorch推理预测耗时： ({t5 - t4:.3f})s.')

                # Apply NMS
                pred_person = non_max_suppression(pred_person, params.conf_thres_person, params.iou_thres_person)
                pred_uniform = non_max_suppression(pred_uniform, params.conf_thres_uniform, params.iou_thres_uniform)
            pass
        pass
    else:
        if data['common_args']['tensorrt'] == True:
            from analysis.trt_inference import trt_infer
            img = preprocess_np(img)
            # Tensorrt Inference
            pred_person = trt_infer(data, engine_person, img)
            pred_uniform = trt_infer(data, engine_uniform, img)
        else:
            img = preprocess_tensor(img, device, half)
            # Pytorch Inference
            t4 = time_synchronized()
            pred_person = MODEL_person(img, augment=False)[0]
            pred_uniform = MODEL_uniform(img, augment=False)[0]
            t5 = time_synchronized()
            logger.info(f'设备 {data["k8sName"]} Pytorch推理预测耗时： ({t5 - t4:.3f})s.')

            # Apply NMS
            t6 = time_synchronized()
            pred_person = non_max_suppression(pred_person, params.conf_thres_person, params.iou_thres_person)
            pred_uniform = non_max_suppression(pred_uniform, params.conf_thres_uniform, params.iou_thres_uniform)
            t7 = time_synchronized()
            logger.info(f'设备 {data["k8sName"]} 推理结果NMS耗时 ({t7 - t6:.3f})s.')
            pass
        pass
    t8 = time_synchronized()
    logger.info(f'设备 {data["k8sName"]} 模型处理图片总耗时 ({t8 - ts:.3f})s.')
    return pred_person, pred_uniform, names_person, names_uniform, img, img_array, degree