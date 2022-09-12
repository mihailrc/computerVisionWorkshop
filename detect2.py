import os
import sys
from pathlib import Path

import cv2

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov7 root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov7') not in sys.path:
    sys.path.append(str(ROOT / 'yolov7'))  # add yolov7 ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolov7_package.detect import Yolov7Detector
from yolov7.utils.datasets import LoadStreams, LoadImages
from yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path

if __name__ == '__main__':

    img2 = cv2.imread('inference/images/bus.jpg')
    dataset = LoadImages('inference/images/bus.jpg')
    # dataset = LoadImages('traffic.mp4')
    print("Loaded:", len(dataset), "images")
    det = Yolov7Detector(weights="yolov7-tiny.pt")
    for path, img, im0s, vid_cap in dataset:
        print(img.shape)
        print(img2.shape)
        print(im0s.shape)
        img3=img.transpose(1,2,0)
        print(img3.shape)
        classes, boxes, scores = det.detect(img3)
        coords=boxes[0]
        print(coords[:, [0, 2]])
        # print(scale_coords(img3.shape, coords, im0s.shape))
        # print(classes, boxes, scores)
        # img = det.draw_on_image(im0s, boxes[0], scores[0], classes[0])
        # cv2.imshow("image", im0s)
        # cv2.waitKey()




    # img = cv2.imread('inference/images/image1.jpg')
    # #img = cv2.resize(img, [640, 640])
    # det = Yolov7Detector(weights="yolov7-tiny.pt")
    # classes, boxes, scores = det.detect(img)
    # print("Found ", len(classes[0]), " objects")
    # print(classes, boxes, scores)
    # img = det.draw_on_image(img, boxes[0], scores[0], classes[0])
    # print("Image shape:", img.shape)
    #
    # cv2.imshow("image", img)
    # cv2.waitKey()
