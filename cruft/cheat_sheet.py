import cv2

vidcap = cv2.VideoCapture("inference/videos/traffic.mp4")
success, image = vidcap.read()
if success:
    cv2.imwrite("first_frame.jpg", image)  # save frame as JPEG f

#perspective transform
#https://medium.com/analytics-vidhya/opencv-perspective-transformation-9edffefb2143