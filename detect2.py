from yolov7_package.detect import Yolov7Detector
from yolov7_package.utils.datasets import LoadStreams, LoadImages
import cv2

if __name__ == '__main__':

    dataset = LoadImages('traffic.mp4')
    print("Loaded:", len(dataset), "images")
    det = Yolov7Detector(weights="yolov7-tiny.pt")
    for path, img, im0s, vid_cap in dataset:
        classes, boxes, scores = det.detect(img)
        img = det.draw_on_image(im0s, boxes[0], scores[0], classes[0])
        cv2.imshow("image", im0s)
        cv2.waitKey(1)




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
