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
from drawUtils import draw_tracking_info
from yolov7.utils.torch_utils import time_synchronized

if __name__ == '__main__':

    dataset = LoadImages('traffic.mp4')
    # dataset = LoadImages('inference/images/bus.jpg')
    print("Loaded:", len(dataset), "images")
    detector = Yolov7Detector(weights="yolov7-tiny.pt", traced=True, classes=[2,3,5,7])
    # detector = Yolov7Detector(weights="yolov7-tiny.pt", traced=True)
    tracker = DeepsSortTracker()
    initializeVideoWriter, vid_writer = False, None
    
    for path, _, im0s, vid_cap in dataset:
        #detection
        t1 = time_synchronized()
        xyxy, scores,class_ids = detector.detect(im0s)
        t2 = time_synchronized()
        print("Detection time (ms):", (t2 - t1) * 1000)
        #tracking
        xyxy_t,class_ids_t, object_ids_t = tracker.update(xyxy, scores, class_ids, im0s)
        t3 = time_synchronized()
        print("Detection time (ms):" , (t2-t1)*1000, " Tracking time(ms): ", (t3-t2)*1000, " Total Time (ms):", (t3-t1)*1000)
               
        if xyxy_t is not None:         
            #draw on images if you wish    
            im0s = detector.draw_boxes(im0s, xyxy_t, scores, class_ids_t, object_ids_t)
            draw_tracking_info(im0s, xyxy_t, class_ids_t, identities=object_ids_t, classes=detector.class_names)

            # if not initializeVideoWriter:  # new video
            #     initializeVideoWriter = True
            #     if isinstance(vid_writer, cv2.VideoWriter):
            #         vid_writer.release()  # release previous video writer
            #
            #     fps, w, h = 30, im0s.shape[1], im0s.shape[0]
            #     vid_writer = cv2.VideoWriter('/workspaces/computerVisionWorkshop/traffictest.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            #
            # vid_writer.write(im0s)

        # if isinstance(vid_writer, cv2.VideoWriter):
       #     vid_writer.release()  # release previous video writer 
        # cv2.imshow("image", im0s)
        cv2.waitKey(1)

