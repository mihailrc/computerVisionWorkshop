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
from yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path

if __name__ == '__main__':

    dataset = LoadImages('inference/images/bus.jpg')
    # dataset = LoadImages('traffic.mp4')
    print("Loaded:", len(dataset), "images")
    detector = Yolov7Detector(weights="yolov7-tiny.pt", traced=True)
    for path, img, im0s, vid_cap in dataset:
        pred = detector.detect(img)
        print("Original:", im0s.shape)
        print("Resized:", img.shape)
        print(pred)
        xyxy_bboxs = []
        confs = []
        oids = []
        for i, det in enumerate(pred):
            # xyxy_bboxs = []
            # confs = []
            # oids = []
            if len(det):
                x_scale=im0s.shape[0]/img.shape[1]
                y_scale=im0s.shape[1]/img.shape[2]
                # Rescale boxes from img_size to im0 size
                # det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in det:
                    coords = torch.tensor(xyxy).tolist()
                    xyxy_scaled = [coords[0] * y_scale, coords[1] * x_scale, coords[2] * y_scale, coords[3] * x_scale]
                    # to deep sort format
                    # x_c, y_c, bbox_w, bbox_h = xyxy2xywh(*xyxy)
                    # xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                    xyxy_bboxs.append(xyxy_scaled)
                    confs.append([conf.item()])
                    # label = '%s' % (names[int(cls)])
                    # color = compute_color_for_labels(int(cls))
                    # UI_box(xyxy, im0, label=label, color=color, line_thickness=2)
                    oids.append(int(cls))
            print(xyxy_bboxs)
            print(confs)
            print(oids)


        # img = detector.draw_on_image(im0s, xyxy_bboxs, confs, oids)
        # cv2.imshow("image", im0s)
        # cv2.waitKey(1)




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
