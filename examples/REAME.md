## Run INference on the potholes video using the trained model best.pt

-   Go to the yolov7 folder by running cd yolov7 and run the below command. THis will run inference on the video and stores the infered video
at /examples/run/detect/exp folder

```
    python detect.py --weights ../examples/best.pt --source ../examples/potholes.mp4 --project ../examples/runs/detect
```