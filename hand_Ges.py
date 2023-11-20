import cv2
import hand_module as hm
import time
import numpy as np 
from keras.models import load_model

model = load_model('Best_points.h5')
letters = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l',
           12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'u', 21: 'v', 22: 'w',
           23: 'x', 24: 'y', 25: 'z', 26: ''}
p_time  ,c_time = 0,0
detector = hm.hand_detect()
prediction = 26
#drawing a black window 
image_canva = np.zeros((500 , 720, 3),np.uint8)
cropped_canvas = image_canva



xp , yp = 0 ,0 
points = []



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

    d =detector.drawxy(img)
    if(d!=0):
          
          x1 , y1 = d
          if xp == 0 and yp == 0:
                 xp, yp = x1, y1
          

          points.append((x1,y1))

          cv2.line(img, (xp, yp), (x1, y1), (0,0,0), 10)
          cv2.line(image_canva, (xp, yp), (x1, y1), (255,255,255), 10)
          xp,yp = x1,y1
          points.append((xp , yp))
    elif(d==0):
          image_canva = np.zeros((500 , 720 , 3),np.uint8)
          points = []
    b = detector.finger_raise(img)

    if(len(points) > 0):
          min_x, min_y = min(points, key=lambda p: p[0])
          max_x, max_y = max(points, key=lambda p: p[0])

          if min_x < max_x and min_y < max_y:
            cropped_canvas = image_canva[min_y-100:max_y+50, min_x-50:max_x+50]
          else:
            cropped_canvas = image_canva
          
    if(d == 0):
          cv2.imwrite('corpes.jpg',cropped_canvas)
   
          gray_frame = cv2.cvtColor(cropped_canvas, cv2.COLOR_BGR2GRAY)

    # Resize the grayscale frame to (28, 28)
          resized_frame = cv2.resize(gray_frame, (28, 28))

    # Add a single channel to create a shape of (28, 28, 1)
          img1 = resized_frame.reshape((28, 28, 1))
          img1 = img1.astype('float32') / 255
          prediction = model.predict(img1.reshape(1, 28, 28))[0]
          prediction = np.argmax(prediction)
          print(letters[int(prediction)] , "\n")
     

    cv2.imshow('MediaPipe Hands',img)
    cv2.imshow('canvas' , image_canva)
    cv2.imshow('cropped' , cropped_canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
	    break



    
          
          
   
#     a = detector.finger_raise(img)
#     if a==2:
#           x1,y1 = lm_list[8][1:]
#           cv2.circle(img , (x1,y1) , 15, color= (0,0,255 , cv2.FILLED))
        
#           cv2.drawMarker(img,(x1,y1), color=(0,255,0), markerType=cv2.MARKER_CROSS, thickness=2)
#           if xp == 0 and yp == 0:
#                 xp, yp = x1, y1

#           cv2.line(img, (xp, yp), (x1, y1), (255,0,0), 10)
#           cv2.line(image_canva, (xp, yp), (x1, y1), (255,0,0), 10)
#           xp,yp = x1,y1
#     if a==5:
#           image_canva = np.zeros((500 , 720 , 3) , np.uint8)
#           image_canva = cv2.rectangle(image_canva,(250,150), (200,230), (255,255,255), -1)


#     if a == 4:
#           cv2.imwrite('img1.png',image_canva) 
#           break
    
cap.release()
cv2.destroyAllWindows()
