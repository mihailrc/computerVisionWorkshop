from pathlib import Path
import os, sys

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
from yolov7.utils.datasets import letterbox
from yolov7.utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from yolov7.utils.torch_utils import select_device, TracedModel

class Yolov7Detector:

    def __init__(self,
                 weights=None,
                 img_size=640,
                 conf_thres=0.25,
                 iou_thres=0.45,
                 augment=False,
                 agnostic_nms=False,
                 device='',
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
        self.img_size=img_size
        self.color = (0, 255, 0)

        # Initialize
        set_logging()
        self.device = select_device(device)
        print("Selected device:", self.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        print("Attempting to load model")
        self.model=attempt_load(weights, map_location=self.device)  # load FP32 model

        self.class_names=self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors=[[np.random.randint(0, 255) for _ in range(3)] for _ in self.class_names]

        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(img_size, s=self.stride)  # check img_size

        if self.traced:
            self.model = TracedModel(self.model, self.device, self.img_size)

        if self.half:
            self.model.half()  # to FP16

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once

        self.old_img_w=self.imgsz
        self.old_img_h=self.imgsz
        self.old_img_b=1

    def detect(self, img0):
        img=self.convert_image(img0,self.imgsz,self.stride)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        #Warmup
        if self.device.type != 'cpu' and (
                self.old_img_b != img.shape[0] or self.old_img_h != img.shape[2] or self.old_img_w != img.shape[3]):
            print("Executing warmup")
            self.old_img_b = img.shape[0]
            self.old_img_h = img.shape[2]
            self.old_img_w = img.shape[3]
            for i in range(3):
                self.model(img, augment=self.augment)[0]

        # Inference
        pred = self.model(img, augment=self.augment)[0]
        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)

        return self.process_predictions(pred, img0, img)

    def convert_image(self, img0, img_size, stride):
        img = letterbox(img0, img_size, stride=stride)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        return img

    def process_predictions(self, pred, im0s, img):
        xyxy_bboxs = []
        scores = []
        class_ids = []
        orig_image_shape = im0s.shape
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orig_image_shape).round()
                for *xyxy, score, cls in det:
                    coords = torch.tensor(xyxy).tolist()
                    xyxy_bboxs.append(coords)
                    scores.append(score.item())
                    class_ids.append(int(cls))
        return xyxy_bboxs, scores, class_ids
