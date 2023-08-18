import cv2
cap1=cv2.VideoCapture('peoplecount1.mp4')
frame=cap1.read()[1]
cv2.imwrite("shop.jpg", frame)
cap1.release()