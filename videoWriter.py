import cv2

class VideoWriter:

    def __init__(self, writeLocation):
        self.writeLocation=writeLocation
        self.video_writer = None

    def write(self, image):
        if self.video_writer is None:
            fps, w, h = 30, image.shape[1], image.shape[0]
            self.video_writer = cv2.VideoWriter(self.writeLocation, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        self.video_writer.write(image)

    def release(self):
        if isinstance( self.video_writer, cv2.VideoWriter):
            self.video_writer.release()
