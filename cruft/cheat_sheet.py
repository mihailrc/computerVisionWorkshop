import cv2

vidcap = cv2.VideoCapture("inference/videos/traffic.mp4")
success, image = vidcap.read()
if success:
    cv2.imwrite("first_frame.jpg", image)  # save frame as JPEG f

#perspective transform
#https://medium.com/analytics-vidhya/opencv-perspective-transformation-9edffefb2143

#for vehicle count per lane
#counter = VehicleCounter(lanes=[[(180, 450),(445, 450), 0],[(450, 450),(650, 450), 0],[(655, 450),(865, 450), 0],[(870, 450),(1100, 450), 0]])

#magic tricks!
                # for i, lane in enumerate(self.lanes):
                #     if self.did_it_cross_the_lane(lane,box,id):
                #         #increment total vehicle count
                #         self.total_count+=1
                #         lane[2] += 1
                #
