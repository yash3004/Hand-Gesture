import cv2
import hand_module as hm
import time
import numpy as np 


p_time  ,c_time = 0,0
detector = hm.hand_detect()
#drawing a black window 
image_canva = np.zeros((500 , 720, 3),np.uint8)
image_canva = cv2.rectangle(image_canva,(100,150), (200,230), (0,0,255), -1)

xp , yp = 0 ,0 



def draw(img , x , y):
      cv2.drawMarker(img, (int(x),int(y)), color=(0,255,0), markerType=cv2.MARKER_CROSS, thickness=2)
      return img

cap = cv2.VideoCapture(0)
while cap.isOpened():
    success , img = cap.read()
    if not success:
        print("camera m pareshanni hai ")
    img = cv2.flip(img , 1)
    img = detector.find_landmark(img , draw = True)
    img = cv2.resize(img , [720,500])
    
    
    detector.find_pos(img)

    c_time = time.time()
    fps = 1/(c_time - p_time)
    p_time = c_time
    cv2.putText(img , str(int(fps)) , (10,90), cv2.FONT_HERSHEY_PLAIN, 3 ,(225,0,0),3)
    lm_list = detector.find_pos(img)
   
    a = detector.finger_raise(img)
    if a==2:
          x1,y1 = lm_list[8][1:]
          cv2.circle(img , (x1,y1) , 15, color= (0,0,255 , cv2.FILLED))
        
          cv2.drawMarker(img,(x1,y1), color=(0,255,0), markerType=cv2.MARKER_CROSS, thickness=2)
          if xp == 0 and yp == 0:
                xp, yp = x1, y1

          cv2.line(img, (xp, yp), (x1, y1), (255,0,0), 10)
          cv2.line(image_canva, (xp, yp), (x1, y1), (255,0,0), 10)
          xp,yp = x1,y1
    if a==5:
          image_canva = np.zeros((500 , 720 , 3) , np.uint8)
          image_canva = cv2.rectangle(image_canva,(250,150), (200,230), (255,255,255), -1)


    if a == 4:
          cv2.imwrite('img1.png',image_canva) 
          break
    cv2.imshow('MediaPipe Hands',img)
    cv2.imshow('canvas' , image_canva)

    if cv2.waitKey(1) & 0xFF == ord('q'):
	    break
cap.release()
cv2.destroyAllWindows()
