import os
import sys
from pathlib import Path

import cv2
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov7 root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov7') not in sys.path:
    sys.path.append(str(ROOT / 'yolov7'))  # add yolov7 ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from detector import Yolov7Detector
from tracker import DeepsSortTracker
from yolov7.utils.datasets import LoadStreams, LoadImages
from drawUtils import draw_boxes

if __name__ == '__main__':

    dataset = LoadImages('traffic.mp4')
    # dataset = LoadImages('inference/images/bus.jpg')
    print("Loaded:", len(dataset), "images")
    detector = Yolov7Detector(weights="yolov7-tiny.pt", traced=True, classes=[2,3,5,7])
    # detector = Yolov7Detector(weights="yolov7-tiny.pt", traced=True)
    tracker = DeepsSortTracker()
    for path, _, im0s, vid_cap in dataset:
        xyxy,xywh, scores,class_ids = detector.detect(im0s)
        outputs = tracker.update(xywh, scores, class_ids, im0s)
        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -2]
            object_id = outputs[:, -1]
            draw_boxes(im0s, bbox_xyxy, object_id, identities)
        # print(outputs)
        # img = detector.draw_boxes(im0s, xyxy, scores, class_ids)
        cv2.imshow("image", im0s)
        cv2.waitKey(1)

