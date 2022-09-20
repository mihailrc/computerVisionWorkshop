import cv2
import torch
import numpy as np

class VehicleCounter:

    def __init__(self, lanes):
        self.counter = 0
        self.cars_id = []
        self.lanes = lanes

    def count(self, xyxy, class_ids, object_ids):

        for i, box in enumerate(xyxy):
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]     

            # get ID of object
            id = int(object_ids[i]) if object_ids is not None else 0       

            x,y = self.center(x1,y1,x2,y2)

            for i, lane in enumerate(self.lanes):
                
                state = self.check_car_position(lane,x,y,id)
                if state:
                    self.counter+=1
                    lane[2]+=1

        return self.counter, self.lanes

    def center(self, x1,y1,x2,y2):
            x = (x1+x2)/2
            y = (y1+y2)/2
            return x,y

    def check_car_position(self,line,x,y,id):
        xLine, yLine, _ = line
        if x> xLine[0] and x < yLine[0]:
            if y > yLine[1]:
                if self.cars_id.__contains__(id):
                    return False
                    
                self.cars_id.append(id)

                return True
            
        return False


