import hand_module as hm 
import cv2 

detector = hm.hand_detect

cap = cv2.VideoCapture(0)
while cap.isOpened():
    success , img = cap.read()
    if not success:
        print("camera m pareshanni hai ")
    img = cv2.flip(img , 1)
    img = detector.find_landmark(img , draw=  True)
    img =  detector.find_pos(img)


    cv2.imshow('MediaPipe Hands',img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
	    break
cap.release()
cv2.destroyAllWindows()
