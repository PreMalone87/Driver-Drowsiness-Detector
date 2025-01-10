import cv2
import mediapipe as mp
import numpy as np
import time
import tensorflow as tf


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence = 0.5, min_tracking_confidence = 0.5)
mp_drawing = mp.solutions.drawing_utils
drwaing_spec = mp_drawing.DrawingSpec(thickness = 1, circle_radius = 1)
model  =  tf.keras.models.load_model("D:\\DriverStateAnalysis\\DriverDrowsinessDetection\\blinkDetectionCode\\CEW.h5")



def findBlink(image):    
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image,(75,75))
    image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    image = image/255.0
    image = [image]
    image = np.array(image)
    pred = model.predict(image)
    if pred[0][0] > pred[0][1]:
        return 0
    print(pred)
    return 1
    


path = "D:\\DriverStateAnalysis\\DriverDrowsinessDetection\\dependencies\\"
face_cascade = cv2.CascadeClassifier(path+'haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencvblob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier(path+'haarcascade_eye.xml')
    



cap = cv2.VideoCapture(0)
yawnCount = 0
blinkCount = 0

while cap.isOpened():
    success, image = cap.read()
    
    if not success:
        continue
    landmarkImage = image.copy()
    start = time.time()
    
    leftEye=""
    rightEye=""
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image.flags.writeable = False
    
    results = face_mesh.process(image)
    
    image.flags.writeable = True
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    img_h, img_w, img_c = image.shape
    
    face_3d= []
    
    face_2d = []
    
    
    
    ##HaarCascade
    gray = cv2.cvtColor(landmarkImage, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.6, minNeighbors = 7, minSize = (30,30))
    eyesDetected = 0
    for (x,y,w,h) in faces:
        #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        #roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            eyesDetected = 1            
           
            break
   
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            #mp.solutions.drawing_utils.draw_landmarks(landmarkImg,face_landmarks) 
            
            imgHeight = image.shape[0]
            imgWidth = image.shape[1]
            
            #right lower corner white box
            
            boxStart = (imgHeight-100,imgWidth-200)  #left top corner of box
            #boxStart = (imgHeight-100,imgWidth-200)  #left top corner of box
            boxEnd = (imgHeight,imgWidth) #right down corner of box
            
            image[boxStart[0] : imgHeight, boxStart[1]:imgWidth] = (255,255,255)
           
            #print("White box cordinates : ", boxStart[1]," ",boxStart[0])
            
            eyes_2d_left = []
            eyes_2d_right = []
            lips = []
            leftEyeIndex = [ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398]
            rightEyeIndex = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246]
            lipsIndex = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
            for idx, lm in enumerate(face_landmarks.landmark):                
               
                if idx in leftEyeIndex:
                    eyes_2d_left.append((int(lm.x*img_w), int(lm.y*img_h)))
                    
                if idx in rightEyeIndex:
                    eyes_2d_right.append((int(lm.x*img_w), int(lm.y*img_h)))
                    
                if idx in lipsIndex:
                    lips.append((int(lm.x*img_w), int(lm.y*img_h)))
                    
                if idx==33 or idx==263 or idx==1 or idx==61 or idx==291 or idx==199:
                    if idx==1:
                        nose_2d = (lm.x*img_w, lm.y*img_h)
                        nose_3d = (lm.x*img_w, lm.y*img_h, lm.z * 3000)
                                   
                    x,y = int(lm.x*img_w), int(lm.y*img_h)
                    #print(x,y)        
                    face_2d.append([x,y])
                    face_3d.append([x,y,lm.z])
                                   
            
            face_2d = np.array(face_2d, dtype = np.float32)
            face_3d = np.array(face_3d, dtype = np.float32)
            #print(type(face_2d[0][0]))
            focal_length = 1*img_w
                                   
            cam_matrix = np.array([[focal_length,0,img_h/2],
                                  [0,focal_length,img_w/2],
                                   [0,0,1]
                                  ], dtype= np.float32)
            
            dist_matrix = np.zeros((4,1), dtype = np.float32)
                                   
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d,face_2d,cam_matrix,dist_matrix)                     
                                   
            rmat, jac = cv2.Rodrigues(rot_vec)
            
            angles, mtxr, mtxq, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
                                   
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360
            
            personLookingDirection = "Looking direction : "
            driverAttentionStatus = "Attention Status : Attentive"
            attentionFlag = 0
            drowsinessStatus = ""
            drowsinessFlag = 0
            
            lookingForwardFlag = 0
                
            if y<-10:
                personLookingDirection += "Left"
            elif x<-10:
                personLookingDirection += "Down"                
                drowsinessStatus += "Driver might be drowsy"
                drowsinessFlag=1
            elif y>10:
                personLookingDirection += "Right"            
            elif x>10:
                personLookingDirection += "Up"
            else:
                personLookingDirection += "Forward"
                lookingForwardFlag=1
            
            if x<-10 or x>10 or y<-13 or y>13:
                driverAttentionStatus = "Attention Status :Not Attentive"
                attentionFlag=1
                
                
            nose_3d_projection , jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
                                   
            p1 = int(nose_2d[0]), int(nose_2d[1])
            p2 = int(nose_2d[0] + y*10), int(nose_2d[1] - x*10)
            print(len(eyes_2d_right))
            x,y,w,h = cv2.boundingRect(np.array(eyes_2d_left))
            x1,y1,w1,h1 = cv2.boundingRect(np.array(eyes_2d_right))
            x2,y2,w2,h2 = cv2.boundingRect(np.array(lips))
            
            
            eyeMouthHeightRatio = h/h2
            if eyeMouthHeightRatio<0.23:
                yawnCount+=1
            else:
                yawnCount=0
            
            leftEye = image[y-15:y+h, x:x+w]
            #rightEye = image[y1-10:y1+h1+10, x1:x1+w1]
            
            leftEyeBlink = findBlink(leftEye)            
            leftEyeBlinkStatus = "Not Blinking"
            if leftEyeBlink:
                leftEyeBlinkStatus = "Blinking"
                blinkCount+=1
            else:
                blinkCount = 0
                
                
            if blinkCount>3 and eyesDetected==1 and lookingForwardFlag==1:
                drowsinessStatus += "Driver might be drowsy"
                drowsinessFlag=1
            elif eyesDetected==1 and drowsinessFlag==1:
                drowsinessFlag=0
                
            '''
            rightEyeBlink = findBlink(rightEye)            
            rightEyeBlinkStatus = "Not Blinking"
            if rightEyeBlink:
                rightEyeBlinkStatus = "Blinking"
            '''
            #cv2.imshow("Right Eye", rightEye)
            
            
            #print(x,y,w,h)
            #print(x1,y1,w1,h1)
            
            image = cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
            image = cv2.rectangle(image,(x1,y1),(x1+w1,y1+h1),(0,0,255),2)
            image = cv2.rectangle(image,(x2,y2),(x2+w2,y2+h2),(0,255,0),2)
            
            #cv2.line(image, p1, p2, (0,255,0),3)
            #fps, driverLookingDirection, driverAttentivenessStatus, 3 solid angles, 5 level drwsiness checkpoint           
            end = time.time()
        
            fps = 1/(end - start+0.0001)
            fps = str(fps)
            
            
            yGap = 17 #10 for space and 15 for fonts
            fontScale = 0.3
            indent = 10              
            
            cv2.putText(image,personLookingDirection,(boxStart[1]+indent,boxStart[0]+15),cv2.FONT_HERSHEY_COMPLEX,fontScale,(0,0,0),1)
            
            cv2.putText(image,"FPS : "+fps,(boxStart[1]+indent,boxStart[0]+15+yGap),cv2.FONT_HERSHEY_COMPLEX,fontScale,(255,0,0),1)
            
            color = (0,255,0)
            if attentionFlag==1:
                color = (0,0,255)
            cv2.putText(image,driverAttentionStatus,(boxStart[1]+indent,boxStart[0]+15+(yGap*2)),cv2.FONT_HERSHEY_COMPLEX,fontScale,color,1)
            
            color = (0,255,155)
            if leftEyeBlinkStatus=="Blinking":
                color = (0,0,255)
            cv2.putText(image,"Blink Status: "+str(blinkCount)+" "+leftEyeBlinkStatus,(boxStart[1]+indent,boxStart[0]+15+(yGap*3)),cv2.FONT_HERSHEY_COMPLEX,fontScale,color,1)
            
            yawnStatus = "Yawn Status : Not Yawning"
            yawnStatusFlag = 0
            if yawnCount > 3:
                yawnStatus = "Yawn Status : Yawning"
                yawnStatusFlag=1
            
            color = (255,175,0)
            if yawnStatusFlag==1:
                color=(0,0,255)
            cv2.putText(image,yawnStatus+" "+str(eyeMouthHeightRatio),(boxStart[1]+indent,boxStart[0]+15+(yGap*4)),cv2.FONT_HERSHEY_COMPLEX,fontScale,color,1)
            
            if drowsinessFlag == 1:
                cv2.putText(image,drowsinessStatus,(boxStart[1]+indent,boxStart[0]+15+(yGap*5)),cv2.FONT_HERSHEY_COMPLEX,fontScale,(0,0,255),1)
                # Add borders with above parameters
                borderWidth = 20
                color = (0,30,255)
                image[0:borderWidth,:] = color
                image[imgHeight-borderWidth : imgHeight,:] = color
                image[:,0:borderWidth] = color
                image[:,imgWidth-borderWidth : imgWidth] = color
            
        
    #cv2.imshow("Landmarks ",landmarkImg)
    cv2.imshow("Driver Drowsiness Detection", image)
    
    if cv2.waitKey(1) & 0xFF ==27:
        break
                                   
cap.release()
cv2.destroyAllWindows()