from pathlib import Path
import os, sys

import cv2
import torch
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov7 root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov7') not in sys.path:
    sys.path.append(str(ROOT / 'yolov7'))  # add yolov7 ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadStreams, LoadImages
from yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

#todo read from coco.yaml
# coco_names = ["person",
#               "bicycle",
#               "car",
#               "motorbike",
#               "aeroplane",
#               "bus",
#               "train",
#               "truck",
#               "boat",
#               "traffic light",
#               "fire hydrant",
#               "stop sign",
#               "parking meter",
#               "bench",
#               "bird",
#               "cat",
#               "dog",
#               "horse",
#               "sheep",
#               "cow",
#               "elephant",
#               "bear",
#               "zebra",
#               "giraffe",
#               "backpack",
#               "umbrella",
#               "handbag",
#               "tie",
#               "suitcase",
#               "frisbee",
#               "skis",
#               "snowboard",
#               "sports ball",
#               "kite",
#               "baseball bat",
#               "baseball glove",
#               "skateboard",
#               "surfboard",
#               "tennis racket",
#               "bottle",
#               "wine glass",
#               "cup",
#               "fork",
#               "knife",
#               "spoon",
#               "bowl",
#               "banana",
#               "apple",
#               "sandwich",
#               "orange",
#               "broccoli",
#               "carrot",
#               "hot dog",
#               "pizza",
#               "donut",
#               "cake",
#               "chair",
#               "sofa",
#               "pottedplant",
#               "bed",
#               "diningtable",
#               "toilet",
#               "tvmonitor",
#               "laptop",
#               "mouse",
#               "remote",
#               "keyboard",
#               "cell phone",
#               "microwave",
#               "oven",
#               "toaster",
#               "sink",
#               "refrigerator",
#               "book",
#               "clock",
#               "vase",
#               "scissors",
#               "teddy bear",
#               "hair drier",
#               "toothbrush"]

class Yolov7Detector:

    def __init__(self,
                 weights=None,
                 img_size=None,
                 conf_thres=0.25,
                 iou_thres=0.45,
                 augment=False,
                 agnostic_nms=False,
                 device='cpu',
                 classes=None,
                 traced=False):

        if img_size is None:
            img_size = 640
        if weights is None:
            weights = ['yolov7.pt']

        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.traced = traced
        self.classes = classes

        # sys.path.append(os.path.join(os.path.dirname(__file__), ""))

        # Initialize
        set_logging()
        self.device = select_device(device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(img_size, s=stride)  # check img_size

        if self.traced:
            self.model = TracedModel(self.model, device, img_size)

        if self.half:
            self.model.half()  # to FP16

        # self.names = coco_names
        self.model.eval()

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(device).type_as(next(self.model.parameters())))  # run once

    def detect(self, img):
        """
        :param x: list of numpy images (e.g. after cv2.imread) or numpy image
        :return: predictions tensor
        """

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        # if self.device.type != 'cpu' and (
        #         old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
        #     old_img_b = img.shape[0]
        #     old_img_h = img.shape[2]
        #     old_img_w = img.shape[3]
        #     for i in range(3):
        #         self.model(img, augment=self.augment)[0]

        # Inference
        # t1 = time_synchronized()

        pred = self.model(img, augment=self.augment)[0]
        # t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
        # t3 = time_synchronized()
        return pred
