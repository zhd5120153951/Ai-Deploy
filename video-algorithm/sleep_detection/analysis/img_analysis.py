from ai_common import setting
from ai_common.img_util import similarity, preprocess_np, preprocess_tensor
from ai_common.log_util import logger
from utils.general import non_max_suppression
from utils.torch_utils import time_sync


def img_preprocess(device, data, model_person, engine, params):
    pred_person = []
    degree = 0
    ts = time_sync()
    img = data['img']  # 前处理后的图
    img_array = data['img_array']  # 原图
    half = False

    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # names_person = MODEL_person.module.names if hasattr(MODEL_person,'module') else MODEL_person.names  # ['person', 'person', 'helmet']
    names_person = ['person', 'head', 'helmet']

    if half:
        model_person.half()

    if data['tools']['keyframe']['key_switch']:
        if data['k8sName'] in setting.last_hist_dic.keys():
            degree, hist = similarity(
                img, setting.last_hist_dic[data['k8sName']])
            if degree != 0 and (degree > float(data['tools']['keyframe']['degree^']) or degree < float(
                    data['tools']['keyframe']['degree_'])):  # 相似度
                t3 = time_sync()
                logger.debug(
                    f'设备 {data["k8sName"]} 相似帧过滤耗时： ({t3 - ts:.3f})s.')
                return pred_person, names_person, img, img_array, degree
            else:
                setting.last_hist_dic[data['k8sName']] = hist

                if data['common_args']['tensorrt'] == True:
                    from analysis.trt_inference import trt_infer
                    img = preprocess_np(img)
                    # Tensorrt Inference
                    pred_person = trt_infer(data, engine, img)
                else:
                    img = preprocess_tensor(img, device, half)
                    # Pytorch Inference
                    t4 = time_sync()
                    pred_person = model_person(img, augment=False)[0]
                    t5 = time_sync()
                    logger.info(
                        f'设备 {data["k8sName"]} Pytorch推理预测耗时： ({t5 - t4:.3f})s.')

                # Apply NMS
                pred_person = non_max_suppression(
                    pred_person, params.conf_thres_person, params.iou_thres_person)
            pass
        pass
    else:
        if data['common_args']['tensorrt'] == True:
            from analysis.trt_inference import trt_infer
            img = preprocess_np(img)
            # Tensorrt Inference
            pred_person = trt_infer(data, engine, img)
        else:
            img = preprocess_tensor(img, device, half)
            # Pytorch Inference
            t4 = time_sync()
            pred_person = model_person(img, augment=False)[0]
            t5 = time_sync()
            logger.info(
                f'设备 {data["k8sName"]} Pytorch推理预测耗时： ({t5 - t4:.3f})s.')
            # Apply NMS
            pred_person = non_max_suppression(
                pred_person, params.conf_thres_person, params.iou_thres_person)
            pass
        pass
    t8 = time_sync()
    logger.info(f'设备 {data["k8sName"]} 模型处理图片总耗时 ({t8 - ts:.3f})s.')
    return pred_person, names_person, img, img_array, degree
