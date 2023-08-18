#import dependencies
import cv2
from ultralytics import YOLO
import supervision as sv
from line_counter import *
#select YOLOv8 small version
model=YOLO('yolov8s.pt')
#fuse model to increase performance and reduce the performed operations
model.fuse()
#dict of class names
CLASS_NAMES_DICT = model.model.names

cap1=cv2.VideoCapture('vehicle-counting.mp4')
#Line start and end Points
LINE_START = sv.Point(-10,350)
LINE_END = sv.Point(1080, 350)

#Box annotation for each detection
box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5)
#create line counter and annotator
line_counter = LineZone(start=LINE_START, end=LINE_END)
line_annotator = LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5,custom_in_text='up',custom_out_text='down')

#intialize the values of enter and exit values
enter=0
exit=0
while True:  
    #read frames  
    ret,frame = cap1.read()
    if ret:
        #resize frame
        frame=cv2.resize(frame,(1020,500))
        if not ret:
            break
        #use YOLO V8 for tracking 
        results=model.track(source=frame, persist=True)
        for result in results:
            #Convert results to supervision detections
            detections = sv.Detections.from_yolov8(result)
            #add tracker id for each detection if exist
            if result.boxes.id is not None:
                detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
        #Add label of tracker id for each box
        labels = [
            f"{tracker_id} {CLASS_NAMES_DICT[class_id]}"
            for _, _,_,class_id, tracker_id
            in detections
        ]
        #annote boxes per frame
        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections,
            labels=labels
        )
        #trigger every detection pass by line enter or exit
        try:
            enter,exit=line_counter.trigger(detections=detections)
        except:
            pass
        line_annotator.annotate(frame=frame, line_counter=line_counter)
    cv2.putText(frame,'Number of cars up= '+str(enter),(20,44),cv2.FONT_HERSHEY_COMPLEX,(1),(0,255,0),2)
    cv2.putText(frame,'Number of cars down= '+str(exit),(20,82),cv2.FONT_HERSHEY_COMPLEX,(1),(0,0,255),2)


    cv2.imshow("RGB", frame)
    #press 'q' to break the loop
    if cv2.waitKey(1) == ord('q'):
        cv2.imwrite("frame_{}.jpg".format(99), frame)
        break

cap1.release()
cv2.destroyAllWindows()

