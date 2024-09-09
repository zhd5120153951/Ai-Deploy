# Plotting utils
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from numpy import unicode

from ai_common.util.general import increment_path
import math
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('TkAgg')

# Settings
matplotlib.rc('font', **{'size': 11})
matplotlib.use('Agg')  # for writing to files only


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


def feature_visualization(x, module_type, stage, n=64):
    """
    x:              Features to be visualized
    module_type:    Module type
    stage:          Module stage within model
    n:              Maximum number of feature maps to plot
    """
    batch, channels, height, width = x.shape  # batch, channels, height, width
    if height > 1 and width > 1:
        project, name = 'runs/features', 'exp'
        save_dir = increment_path(Path(project) / name)  # increment run
        save_dir.mkdir(parents=True, exist_ok=True)  # make dir

        plt.figure(tight_layout=True)
        blocks = torch.chunk(x, channels, dim=1)  # block by channel dimension
        n = min(n, len(blocks))
        for i in range(n):
            feature = transforms.ToPILImage()(blocks[i].squeeze())
            ax = plt.subplot(int(math.sqrt(n)), int(math.sqrt(n)), i + 1)
            ax.axis('off')
            plt.imshow(feature)  # cmap='gray'

        f = f"stage_{stage}_{module_type.split('.')[-1]}_features.png"
        print(f'Saving {save_dir / f}...')
        plt.savefig(save_dir / f, dpi=300)


def plot_one_box(x, im, label=None, color=(128, 128, 128),  line_thickness=3):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(
        0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    return im
