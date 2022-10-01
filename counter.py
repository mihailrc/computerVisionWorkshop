class VehicleCounter:

    def __init__(self, lanes):
        self.total_count = 0
        self.cars_id = []
        self.lanes = lanes

    def count(self, xyxy, class_ids, object_ids):

        for i, box in enumerate(xyxy):

            # get ID of object
            id = int(object_ids[i]) if object_ids is not None else 0

            if self.did_it_cross_the_lane(self.lanes[0], box, id):
                self.total_count += 1

        return self.total_count, self.lanes

    def center(self, box):
            x = (box[0]+box[2])/2
            y = (box[1]+box[3])/2
            return x,y

    def did_it_cross_the_lane(self,lane,box,id):
        startPoint, endPoint, _ = lane
        x,y=self.center(box)
        #check if is on correct lane
        if x> startPoint[0] and x < endPoint[0]:
            #check if it crossed the lane
            if y > endPoint[1]:
                # ignore if we already counted this car
                if self.cars_id.__contains__(id):
                    return False
                    
                #keep track of objects that crossed the lane
                self.cars_id.append(id)

                return True
            
        return False


