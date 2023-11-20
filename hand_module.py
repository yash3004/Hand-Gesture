import cv2 
import mediapipe as mp 
import time 

mp_drawing_styles = mp.solutions.drawing_styles

class hand_detect():
    def __init__(self ,mode = False,max_hands = 2,model_com = 0, min_dec_con = 0.5 , trackCon = 0.5):
        self.mode = mode 
        self.model_com = model_com
        self.min_dec_conf = min_dec_con
        self.max_hands = max_hands
        self.trackCon = trackCon
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(self.mode , self.max_hands , self.model_com,self.min_dec_conf , self.trackCon)

    def find_landmark(self,img , draw = False):
        
        
        
        image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.results = self.hands.process(image)
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:

                if draw:
                    self.mp_draw.draw_landmarks(
                        image,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS)
                    
        return image
    def find_pos(self, img , handno=0, draw = True):
         lm_list = []
         if self.results.multi_hand_landmarks:
             my_hand = self.results.multi_hand_landmarks[handno]
             for id,lm in enumerate(my_hand.landmark):
                 
                 h,w,c = img.shape
                 cx , cy = int(lm.x*w) ,int(lm.y*h)
                 lm_list.append([id , cx , cy])
                 print()
         return lm_list
    
    def finger_raise(self,img):
        fingers = []
        lms_list = self.find_pos(img)
        if len(lms_list)!=0:
            if(lms_list[4][1] > lms_list[3][1]):
                fingers.append(1)
            else :
                fingers.append(0)
            

            tip_id = [4,8,12,16,20]
            for i in range(1,5):
                if(lms_list[tip_id[i]][2] < lms_list[tip_id[i]-2][2]):
                    fingers.append(1)
                else:
                    fingers.append(0)
                
            
        a = fingers.count(1)
        return a
    def drawxy(self,img):
        if self.results.multi_hand_landmarks:
            for landmarks in self.results.multi_hand_landmarks:
                # You can process the landmarks to detect hand gestures
                # For a simple example, we will calculate the distance between the thumb and index finger
                thumb_tip = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                index_finger_tip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]

                distance = cv2.norm(
                    (thumb_tip.x, thumb_tip.y), (index_finger_tip.x, index_finger_tip.y)
                )

                if distance < 0.05:
                    h , w , c = img.shape
                    x1,y1 = int(index_finger_tip.x*w) , int(index_finger_tip.y*h)
                    return (x1,y1)

        return 0



def main():

    if __name__ == "__main__": 
        main()