from ai_common import setting
from ai_common.img_util import base64_to_img
from ai_common.log_util import logger


def images_queue_get():
    capture_dic = setting.image_queue.get()
    if setting.TEST_FLAG == 0:
        if setting.image_queue.qsize() > 30:
            setting.image_queue.queue.clear()
        if capture_dic['img_str'] is not None:
            img_from_base64, h0w0, img_array = base64_to_img(capture_dic['img_str'])
            capture_dic['img'] = img_from_base64
            capture_dic['h0w0'] = h0w0
            capture_dic['img_array'] = img_array
        else:
            raise Exception('图片内容异常')
    logger.info(f"image_queue.qsize():{setting.image_queue.qsize()}; -> Get images form the image_queue -> image_queue.full():{setting.image_queue.full()}\n")

    return capture_dic


def rtsp_queue_get():
    capture_dic = setting.image_queue.get()
    if setting.TEST_FLAG == 0:
        if setting.image_queue.qsize() > 30:
            setting.image_queue.queue.clear()
        if capture_dic['img_array'] is None and capture_dic['img'] is None:
            raise Exception('图片内容异常')
    logger.info(
        f"image_queue.qsize():{setting.image_queue.qsize()}; -> Get images form the image_queue -> image_queue.full():{setting.image_queue.full()}\n")

    return capture_dic