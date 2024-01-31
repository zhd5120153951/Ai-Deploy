'''
@FileName   :camera.py
@Description:单独从utils/dataloaders.py中提取出数据加载--精简后的
@Date       :2024/01/30 08:46:19
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
import os
import cv2
import glob
import time
import numpy as np
from pathlib import Path
from utils.augmentations import letterbox
from threading import Thread
from utils.general import clean_str, is_colab, is_kaggle, LOGGER


img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif',
               'tiff', 'dng', 'webp']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg',
               'm4v', 'wmv', 'mkv']  # acceptable video suffixes


class LoadImages:  # for inference
    def __init__(self, path, img_size=640, stride=32):
        p = str(Path(path).absolute())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        videos = [x for x in files if x.split('.')[-1].lower() in vid_formats]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {img_formats}\nvideos: {vid_formats}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print(
                f'video {self.count + 1}/{self.nf} ({self.frame}/{self.nframes}) {path}: ', end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
            print(f'image {self.count}/{self.nf} {path}: ', end='')

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files


class LoadWebcam:  # for inference
    def __init__(self, pipe='0', img_size=640, stride=32):
        self.img_size = img_size
        self.stride = stride

        if pipe.isnumeric():
            pipe = eval(pipe)  # local camera
        # pipe = 'rtsp://192.168.1.64/1'  # IP camera
        # pipe = 'rtsp://username:password@192.168.1.64/1'  # IP camera with login
        # pipe = 'http://wmccpinetop.axiscam.net/mjpg/video.mjpg'  # IP golf camera

        self.pipe = pipe
        self.cap = cv2.VideoCapture(pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        if self.pipe == 0:  # local camera
            ret_val, img0 = self.cap.read()
            img0 = cv2.flip(img0, 1)  # flip left-right
        else:  # IP camera
            n = 0
            while True:
                n += 1
                self.cap.grab()
                if n % 30 == 0:  # skip frames
                    ret_val, img0 = self.cap.retrieve()
                    if ret_val:
                        break

        # Print
        assert ret_val, f'Camera Error {self.pipe}'
        img_path = 'webcam.jpg'
        print(f'webcam {self.count}: ', end='')

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return img_path, img, img0, None

    def __len__(self):
        return 0


class LoadStreams:  # multiple IP or RTSP cameras
    # YOLOv5 streamloader, i.e. `python detect.py --source 'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP streams`
    def __init__(self, sources='streams.txt', img_size=640, stride=32, auto=True, transforms=None, vid_stride=1):
        # torch.backends.cudnn.benchmark=True#原始写法
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride
        self.vid_stride = vid_stride  # 跳帧间隔

        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read(
                ).strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs = [None] * n  # 读取帧
        self.threads = [None]*n  # 读取线程
        # 清理流地址中的非法字符
        self.sources = [clean_str(x) for x in sources]
        for i, s in enumerate(sources):
            # Start the thread to read frames from the video stream
            print(f'{i + 1}/{n}: {s}... ', end='')
            s = eval(s) if s.isnumeric() else s  # '0'--本地摄像头
            if s == 0:
                assert not is_colab(), '--source 0 webcam unsupported on Colab. Rerun command in a local environment.'
                assert not is_kaggle(
                ), '--source 0 webcam unsupported on Kaggle. Rerun command in a local environment.'
            cap = cv2.VideoCapture(s)
            assert cap.isOpened(), f'Failed to open {s}'
            # 获取流视频的宽高信息
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) % 100  # 控制在100以内,原始的不是这样写的

            _, self.imgs[i] = cap.read()  # guarantee first frame
            self.threads[i] = Thread(target=self.update, args=(
                [i, cap, s]), daemon=True)
            # print(f' success ({w}x{h} at {fps:.2f} FPS).')#zhd
            LOGGER.info(f" success ({w}*{h} at {fps:.2f} FPS).")  # origin
            self.threads[i].start()
        # print('')  # newline zhd
        LOGGER.info('')  # newline origin
        # check for common shapes
        s = np.stack([letterbox(x, self.img_size, stride=self.stride, auto=auto)[
                     0].shape for x in self.imgs], 0)  # shapes--origin no 0
        # rect inference if all shapes equal
        self.rect = np.unique(s, axis=0).shape[0] == 1
        self.auto = auto and self.rect
        self.transforms = transforms  # 可选
        if not self.rect:
            # print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')#zhd
            LOGGER.warn(
                'WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')  # origin

    def update(self, index, cap, stream):
        # Read next stream frame in a daemon thread
        n = 0  # frame number
        while cap.isOpened():
            n += 1
            cap.grab()  # read()=grab() followed by retrieve()
            if n % self.vid_stride == 0:  # read every vid_stride frame
                success, im = cap.retrieve()
                if success:
                    self.imgs[index] = im
                else:
                    LOGGER.warn(
                        'WARNING ⚠️ Video stream unresponsive, please check your IP camera connection.')
                    self.imgs[index] = np.zeros_like(
                        self.imgs[index])  # origin
                    # self.imgs[index] = self.imgs[index] * 0#zhd
                    # re-open stream if signal was lost
                    cap.open(stream)
                n = 0
            time.sleep(0.0)  # wait time zhd-0.01;origin-0.0

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        # origin
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration
        # zhd
        # if cv2.waitKey(1) == ord('q'):  # q to quit
        #     cv2.destroyAllWindows()
        #     raise StopIteration

        # origin
        img0 = self.imgs.copy()
        if self.transforms:  # 默认不开启的
            img = np.stack([self.transforms(x) for x in img0])
        else:
            img = np.stack([letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[
                           0] for x in img0])  # resize
            # BGR to RGB ,BHWC to BCHW
            img = img[..., ::-1].transpose((0, 3, 1, 2))
            img = np.ascontiguousarray(img)  # contiguous
        return self.sources, img, img0, None, ''

        # zhd
        # Letterbox-1
        img = [letterbox(x, self.img_size, auto=self.rect,
                         stride=self.stride)[0] for x in img0]
        # Stack-2
        img = np.stack(img, 0)

        # Convert-3
        # BGR to RGB, to bsx3x416x416
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None

    def __len__(self):
        return len(self.sources)  # origin
        # zhd
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years
