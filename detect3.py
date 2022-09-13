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
from yolov7.utils.datasets import LoadStreams, LoadImages

if __name__ == '__main__':

    # dataset = LoadImages('inference/images/bus.jpg')
    dataset = LoadImages('traffic.mp4')
    print("Loaded:", len(dataset), "images")
    detector = Yolov7Detector(weights="yolov7-tiny.pt", traced=True)
    for path, _, im0s, vid_cap in dataset:
        xxyy,scores,class_ids = detector.detect(im0s)
        img = detector.draw_boxes(im0s, xxyy, scores, class_ids)
        cv2.imshow("image", img)
        cv2.waitKey(1)

