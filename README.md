# Object Detection,Tracking and Counting
## YOLOV8 +SUPERVISION

## Steps
1-Download the videos from https://drive.google.com/drive/folders/1Xad2Wo2AyccTh-itYRq_pWujDwGQSGHJ
2-use cap_frame.py to save a frame from a video 
3- Add captured frame to  https://roboflow.github.io/polygonzone/  to draw the line and save the annotations of the points
4- Modify the trigger function from line_counter.py from Supervision  to return self.in_count and self.out_count
5-In project file(car counter.py / peoplecounter.py)
Download the YOLO V8 small model and fuse it
6-Read the video 
7- Create the Line and Box annotations
8-iterate over each frame of the video
9-using model.track to detect and track each object in the frame
10-Convert detections to sv Detections
11-Add id for each detection if there are any detection
12- Filter sv.detections by class or confidence if needed 
13-create label of each bbox [I choose to label by tracker id only]
14-Save the values of the two sides (enter and exit) from trigger function
15-Using try and except ? Because the function will try None if there function trigger nothing
16- use line.annotator to show the line on the video
17- Add text to identify the counters
18-show the video using cv2.show
19-Press 'q' to break the loop and close the video

N.B:
Video 'peoplecount1.mp4' -----> Used in Project file  'people Counter.py'
video 'vehicle-counting.mp4' ------> Used in Project file 'car counter.py'


## How To Use 
1-Clone the repo
2-Download the videos in the directory of the repo
3-Use cap_frame.py to save a frame from video as image
4-use polygon zone https://roboflow.github.io/polygonzone/ to draw the line and save the annotations
5- update the line points in the project file
6- run the project file.

