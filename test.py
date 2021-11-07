
import cv2 


cap = cv2.VideoCapture('rtsp://admin:QSCUVQ@192.168.0.104:554/Streaming/Channels/2')
weight = 1280
height = 720
while True:
    grabbed, frame = cap.read()
    frame = cv2.resize(frame,(weight,height))
    cv2.imshow('video',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()