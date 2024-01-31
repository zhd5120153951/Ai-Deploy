# import the necessary packages
from yolov5 import YOLONet
from camera import LoadStreams, LoadImages
from utils.general import non_max_suppression, scale_boxes, check_imshow
from flask import Response
from flask import Flask
from flask import render_template
import time
import torch
import json
import cv2


# initialize a flask object
app = Flask(__name__)

# initialize the video stream and allow the camera sensor to warmup
with open('yolov5_config.json', 'r', encoding='utf8') as fp:
    opt = json.load(fp)
    print('[INFO] YOLOv5 Config:', opt)

yolo = YOLONet(opt)
if yolo.webcam:
    # cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(
        yolo.source, yolo.imgsz, yolo.stride, False, vid_stride=yolo.vid_stride)
else:
    dataset = LoadImages(
        yolo.source, yolo.imgsz, yolo.stride)
# time.sleep(2.0)


@app.route("/")
def index():
    return render_template("index.html")


def detect_gen(yolo, dataset, feed_type):
    view_img = check_imshow(True)
    for path, img, img0s, vid_cap, _ in dataset:
        img, img0 = yolo._preprocess(img)

        pred = yolo.model(img, yolo.augment)[0]  # 0.22s
        pred = pred.float()
        pred = non_max_suppression(
            pred, yolo.conf_thresh, yolo.iou_thresh)

        # pred_boxes = []
        for i, det in enumerate(pred):
            if yolo.webcam:  # batch_size >= 1
                feed_type_curr, p, s, im0, frame = "Camera_%s" % str(
                    i), path[i], '%g: ' % i, img0s[i].copy(), dataset.count
            else:
                feed_type_curr, p, s, im0, frame = "Camera_", path, '', img0s, getattr(
                    dataset, 'frame', 0)

            s += '%gx%g ' % img.shape[2:]  # print string
            # normalization gain whwh
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if det is not None and len(det):
                det[:, :4] = scale_boxes(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results--后台服务可不要
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # add to string
                    s += f"{n} {yolo.names[int(c)]}{'s' * (n > 1)}, "

                for *xyxy, conf, cls_id in reversed(det):
                    lbl = '' if yolo.hide_labels else yolo.names[int(cls_id)]
                    xyxy = torch.tensor(xyxy).view(1, 4).view(-1).tolist()
                    score = round(conf.tolist(), 3)
                    label = "{}".format(
                        lbl) if yolo.conf_thresh else "{}: {}".format(lbl, score)
                    x1, y1, x2, y2 = int(xyxy[0]), int(
                        xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    # pred_boxes.append((x1, y1, x2, y2, lbl, score))
                    if view_img:
                        yolo._plot_one_box(
                            xyxy, im0, color=(0, 255, 0), label=label)

            # Print time (inference + NMS)
            # print(pred_boxes)
            if feed_type_curr == feed_type:
                frame = cv2.imencode('.jpg', im0)[1].tobytes()
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed/<feed_type>')
def video_feed(feed_type):
    """Video streaming route. Put this in the src attribute of an img tag."""
    if feed_type == 'Camera_0':
        return Response(detect_gen(yolo=yolo, dataset=dataset, feed_type=feed_type),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    elif feed_type == 'Camera_1':
        return Response(detect_gen(yolo=yolo, dataset=dataset, feed_type=feed_type),
                        mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':

    app.run(host='0.0.0.0', port="5000", threaded=True)
