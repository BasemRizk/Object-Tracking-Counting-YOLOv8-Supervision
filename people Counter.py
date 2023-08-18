#Import Dependencies
import cv2
from ultralytics import YOLO
import supervision as sv
from line_counter import *
#select YOLOv8 small version
model=YOLO('yolov8s.pt')
#fuse model to increase performance and reduce the performed operations
model.fuse()
cap1=cv2.VideoCapture('peoplecount1.mp4')
#Line start and end Points
LINE_START = sv.Point(375,34)
LINE_END = sv.Point(375, 480)

#Box annotation for each detection
box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5)
#create line counter and annotator
line_counter = LineZone(start=LINE_START, end=LINE_END)
line_annotator = LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5,custom_in_text='enter',custom_out_text='exit')

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
                #select detections of person class only
                detections = detections[detections.class_id == 0]
        #Add label of tracker id for each box
        labels = [
            f"{tracker_id} "
            for _, _,_, _, tracker_id
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
        '''Remove the comment to view the Line of  detection'''
        #line_annotator.annotate(frame=frame, line_counter=line_counter)
    cv2.putText(frame,'Number of entering people= '+str(enter),(20,44),cv2.FONT_HERSHEY_COMPLEX,(1),(0,255,0),2)
    cv2.putText(frame,'Number of exiting people= '+str(exit),(20,82),cv2.FONT_HERSHEY_COMPLEX,(1),(0,0,255),2)
        

    cv2.imshow("RGB", frame)
    #press 'q' to get out of the loop
    if cv2.waitKey(1) == ord('q'):
        break

cap1.release()
cv2.destroyAllWindows()

