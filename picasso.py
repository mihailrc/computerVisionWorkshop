import cv2
import numpy as np
from yolov7.utils.plots import plot_one_box
from collections import deque

class Picasso:
    def __init__(self,
                 class_names,
                 colors):
        self.class_names=class_names
        self.colors=colors
        self.tracked_objects={}
        self.buffer_length=64

    def draw_detection_boxes(self, img, xyxy, scores, class_ids):
        for i, box in enumerate(xyxy):
            label = "{class_name:}: {score:.2f}".format(class_name=self.class_names[int(class_ids[i])], score=scores[i])
            plot_one_box(box, img, label=label, color=self.colors[int(class_ids[i])], line_thickness=1)
        return img

    def draw_counter_info(self, img, lanes):
        #green
        color=(0,255,0)
        if len(lanes)==1:
            cv2.putText(img, f"Vehicle Count: {lanes[0][2]}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.line(img, lanes[0][0], lanes[0][1], color, 4)
        else:
            for i, lane in enumerate(lanes):
                cv2.putText(img, f"Lane {(i+1)} Count: {lane[2]}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.line(img, lane[0], lane[1], color, 4)

    def process_tracked_object(self, identities, boxes):
        # remove tracked point from buffer if object is lost
        for key in list(self.tracked_objects):
            if key not in identities:
                self.tracked_objects.pop(key)

        for i, box in enumerate(boxes):
            # get ID of object
            id = int(identities[i]) if identities is not None else 0

            # create new buffer for new object
            if id not in self.tracked_objects:
                self.tracked_objects[id] = deque(maxlen=self.buffer_length)

            center=(int((box[0]+box[2])/2), int((box[1] + box[3])/2))
            self.tracked_objects[id].appendleft(center)

        return self.tracked_objects

    def draw_tracking_info(self, img, boxes, classes, identities):
        self.process_tracked_object(identities,boxes)
        for i, box in enumerate(boxes):
            label = "{class_name:}: {indentity:}".format(class_name=self.class_names[classes[i]], indentity=identities[i])
            plot_one_box(box, img, label=label, color=self.colors[int(classes[i])], line_thickness=1)
            id = int(identities[i]) if identities is not None else 0
            self.draw_trail(classes[i], id, img)

    def draw_trail(self, class_id, id, img):
        for i in range(1, len(self.tracked_objects[id])):
            # check if on buffer value is none
            if self.tracked_objects[id][i - 1] is None or self.tracked_objects[id][i] is None:
                continue
            # generate dynamic thickness of trails
            thickness = int(np.sqrt(self.buffer_length / float(i + i)) * 1.5)
            # draw trails
            cv2.line(img, self.tracked_objects[id][i - 1], self.tracked_objects[id][i],
                     self.colors[int(class_id)], thickness)