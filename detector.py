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
from yolov7.utils.datasets import LoadStreams, LoadImages, letterbox
from yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

class Yolov7Detector:

    def __init__(self,
                 weights=None,
                 img_size=640,
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
        self.img_size=img_size
        self.cars_id = []
        self.color = (0, 255, 0)
        self.line = (180, 450),(1100, 450)
        self.counter = 0

        # sys.path.append(os.path.join(os.path.dirname(__file__), ""))

        # Initialize
        set_logging()
        self.device = select_device(device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(weights, map_location=device)  # load FP32 model
        self.class_names=self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors=[[np.random.randint(0, 255) for _ in range(3)] for _ in self.class_names]

        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(img_size, s=self.stride)  # check img_size

        if self.traced:
            self.model = TracedModel(self.model, device, self.img_size)

        if self.half:
            self.model.half()  # to FP16

        # # self.names = coco_names
        # self.model.eval()
        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(device).type_as(next(self.model.parameters())))  # run once

    def detect(self, img0):
        """
        :param x: list of numpy images (e.g. after cv2.imread) or numpy image
        :return: predictions tensor
        """

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]
        orig_image_shape=img0.shape
        resized_img_shape=img.shape

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

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

        xyxy_bboxs = []
        scores = []
        class_ids = []
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orig_image_shape).round()
                for *xyxy, score, cls in det:
                    coords = torch.tensor(xyxy).tolist()
                    xyxy_bboxs.append(coords)
                    scores.append(score.item())
                    class_ids.append(int(cls))

        return xyxy_bboxs, scores, class_ids

    def draw_boxes(self, img, xyxy, scores, class_ids):
        for i, box in enumerate(xyxy):
            label="{class_name:}: {score:.2f}".format(class_name=self.class_names[int(class_ids[i])], score=scores[i])
            plot_one_box(box, img, label=label, color=self.colors[int(class_ids[i])], line_thickness=1)
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]            

            x,y = self.center(x1,y1,x2,y2)
            state = self.check_car_position(x,y,id)
            if state:
                self.counter+=1

            cv2.putText(img, f"Total Cars crossed: {self.counter}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, self.color, 2)        
            cv2.line(img,self.line[0],self.line[1], self.color,4)   
        return img

    def center(self, x1,y1,x2,y2):
            x = (x1+x2)/2
            y = (y1+y2)/2
            return x,y

    def check_car_position(self,x,y,id):
        xLine, yLine = self.line
        if x> xLine[0] and x < yLine[0]:
            if y > yLine[1] and ((y - yLine[1]) <= 32):
                return True
            
        return False