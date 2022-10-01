import argparse

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

from yolov7.utils.datasets import LoadImages
from yolov7.utils.torch_utils import time_synchronized
from detector import Yolov7Detector
from tracker import DeepsSortTracker
from counter import VehicleCounter
from picasso import Picasso
from videoWriter import VideoWriter

def detect(opt):

    source, weights, view_img, classes, trace = opt.source, opt.weights, opt.view_img, opt.classes, not opt.no_trace

    dataset = LoadImages(source)
    detector = Yolov7Detector(weights=weights, traced=trace, classes=classes)
    tracker = DeepsSortTracker()
    counter = VehicleCounter(lanes=[[(180, 450),(1100, 450), 0]])
    #for vehicle count per lane
    # counter = VehicleCounter(lanes=[[(180, 450),(445, 450), 0],[(450, 450),(650, 450), 0],[(655, 450),(865, 450), 0],[(870, 450),(1100, 450), 0]])
    picasso = Picasso(class_names=detector.class_names, colors=detector.colors)
    videoWriter = VideoWriter(writeLocation="testTraffic.mp4")

    for path, _, im0s, vid_cap in dataset:
        #detection
        t1 = time_synchronized()
        xyxy, scores,class_ids = detector.detect(im0s)
        #draw detection boxes
        # picasso.draw_detection_boxes(im0s,xyxy,scores,class_ids)
        t2 = time_synchronized()
        # print("Detection time (ms):" , (t2-t1)*1000)
        #tracking
        xyxy_t,class_ids_t, object_ids_t = tracker.update(xyxy, scores, class_ids, im0s)
        t3 = time_synchronized()
        print("Detection time (ms):" , (t2-t1)*1000, " Tracking time(ms): ", (t3-t2)*1000, " Total Time (ms):", (t3-t1)*1000)
        if xyxy_t is not None:
            #draw tracking info if you wish
            picasso.draw_tracking_info(im0s,xyxy_t,class_ids_t,object_ids_t)
            # count vehicles
            counter.count(xyxy_t, class_ids_t, object_ids_t)
            picasso.draw_counter_info(im0s,counter)

        if view_img:
            cv2.imshow("image", im0s)
            cv2.waitKey(1)
        else:
            videoWriter.write(im0s)

    videoWriter.release()

    print("Total Vehicle Count:", counter.total_count)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)    

    with torch.no_grad():
        detect(opt)
