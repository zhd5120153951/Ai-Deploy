# -*- coding: utf-8 -*-
"""
@Time ： 2023/6/30 14:21
@Auth ： zhangliang
@File ：multiprocessing_test_onnx.py
"""

import cv2
import time
import re
import multiprocessing as mp
import numpy as np
import onnxruntime as ort


my_labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
             'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
             'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
             'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
             'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
             'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
             'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
             'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
             'hair drier', 'toothbrush']


class Yolov8ONNXModel:

    def __init__(self, onnx_model, confidence_thres, iou_thres):
        """
        Initializes an instance of the Yolov8 class.

        Args:
            onnx_model: Path to the ONNX model.
            input_image: Path to the input image.
            confidence_thres: Confidence threshold for filtering detections.
            iou_thres: IoU (Intersection over Union) threshold for non-maximum suppression.
        """
        self.onnx_model = onnx_model
        # self.input_image = input_image
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        # Load the class names from the COCO dataset
        self.classes = my_labels

        # Generate a color palette for the classes
        self.color_palette = np.random.uniform(
            0, 255, size=(len(self.classes), 3))

        # Create an inference session using the ONNX model and specify execution providers
        self.session = ort.InferenceSession(self.onnx_model, providers=[
                                            'CUDAExecutionProvider', 'CPUExecutionProvider'])

        # Get the model inputs
        self.model_inputs = self.session.get_inputs()

        # Store the shape of the input for later use
        input_shape = self.model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]

    def draw_detections(self, img, box, score, class_id):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """

        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)),
                      (int(x1 + w), int(y1 + h)), color, 2)

        # Create the label text with class name and score
        label = f'{self.classes[class_id]}: {score:.2f}'

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color,
                      cv2.FILLED)

        # Draw the label text on the image
        cv2.putText(img, label, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def preprocess(self, cv2_img):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        # Read the input image using OpenCV
        # self.img = cv2.imread(self.input_image)
        self.img = cv2_img

        # Get the height and width of the input image
        self.img_height, self.img_width = self.img.shape[:2]

        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        # Resize the image to match the input shape
        img = cv2.resize(img, (self.input_width, self.input_height))

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # Return the preprocessed image data
        return image_data

    def postprocess(self, input_image, output):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The output of the model.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """

        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output[0]))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= self.confidence_thres:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(
            boxes, scores, self.confidence_thres, self.iou_thres)

        # Iterate over the selected indices after non-maximum suppression
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]

            # Draw the detection on the input image
            self.draw_detections(input_image, box, score, class_id)

        # Return the modified input image
        return input_image

    # def load_model(self):

    def predict(self, cv2_img):

        # Preprocess the image data
        img_data = self.preprocess(cv2_img)

        # Run inference using the preprocessed image data
        outputs = self.session.run(None, {self.model_inputs[0].name: img_data})

        # Perform post-processing on the outputs to obtain output image.
        output_img = self.postprocess(self.img, outputs)

        # Return the resulting output image
        return output_img

# 先检测后再放到队列中,然后队列拿出来播放--结合前面的再开一个进程,单独拉取流


def image_put(q, rtsp_url, model_path):
    # model = YOLO(model_path)
    onnxModel = Yolov8ONNXModel(model_path, 0.51, 0.45)

    cap = cv2.VideoCapture(rtsp_url)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('fps: ', fps)
    if cap.isOpened():
        print('cap.isOpened')
    else:
        cap = cv2.VideoCapture(rtsp_url)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print('fps: ', fps)
     # 返回当前时间
    start_time = time.time()
    counter = 0
    while cap.isOpened():
        # print('cap.read()[0]:', cap.read()[0])
        ret, frame = cap.read()

        if ret and frame is not None:
            frame = onnxModel.predict(frame)

        counter += 1  # 计算帧数
        if (time.time() - start_time) != 0:
            cv2.putText(frame, "FPS:{0}".format(float('%.1f' % (counter / (time.time() - start_time)))), (5, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # print('ret:', ret)
        frame = cv2.resize(frame, (1920, 1080))
        if not ret:
            cap = cv2.VideoCapture(rtsp_url)
            # print('HIKVISION2')
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1920, 1080))
        q.put(frame)
        # print('q.qsize():', q.qsize())
        q.get() if q.qsize() > 1 else time.sleep(0.01)


def image_get(q, window_name, save_dir):
    ip_strs = re.findall(r"@(.+?):554", window_name)
    ip_str = ip_strs[0]
    time_str = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # path = save_dir + ip_str + "_" +time_str +".avi"
    # out = cv2.VideoWriter(path, fourcc, 20.0, (1920, 1080), True)
    while True:
        frame = q.get()
        # out.write(frame)
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)


def run_multi_camera(rtsp_urls, model_path, save_dir):
    mp.set_start_method(method='spawn')  # init
    queues = [mp.Queue(maxsize=2) for _ in rtsp_urls]

    processes = []
    for queue, rtsp_url in zip(queues, rtsp_urls):
        processes.append(mp.Process(target=image_put,
                         args=(queue, rtsp_url, model_path)))
        processes.append(mp.Process(target=image_get,
                         args=(queue, rtsp_url, save_dir)))

    for process in processes:
        process.daemon = True
        process.start()
    for process in processes:
        process.join()


if __name__ == '__main__':
    # 可以放到文件中配置
    rtsp_urls = [
        "rtsp://admin:jiankong123@192.168.23.10:554/Streaming/Channels/101",
        "rtsp://admin:jiankong123@192.168.23.15:554/Streaming/Channels/101"
    ]
    save_dir = "data/"
    model_path = "models/yolov8s.onnx"
    run_multi_camera(rtsp_urls, model_path, save_dir)
