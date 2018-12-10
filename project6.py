# Name: Asra Saeed
# AndrewID: asras
# Project: filters

import cv2
import Tkinter
from PIL import ImageTk, Image

# load cascade to use
# I had to download the xml files on my computer,
# and link them to cascadeclassifiers to load them

face_cascade = cv2.CascadeClassifier(r"C:\Users\asras\Desktop\uni\So-S1\Python\Project\data\haarcascades\haarcascade_frontalface_alt.xml")
eye_cascade = cv2.CascadeClassifier(r"C:\Users\asras\Desktop\uni\So-S1\Python\Project\data\frontalEyes35x16.xml")

video_capture = cv2.VideoCapture(0)

while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        glasses = cv2.imread('sunglasses.png')

        for (x,y,w,h) in faces:
                # store face coordinates, width and height
                faceBox = (x,y,w,h)
                # store a box with just the face
                # one stored in grayscale and one in normal form
                faceLocate = frame[y:y+h, x:x+w]
                faceLocate2 = gray[y:y+h, x:x+w]
                        
                #______SUNGLASSES FILTER________________

                # detect eyes in face using cascade
                eyes = eye_cascade.detectMultiScale(faceLocate2)
                # loop through eye coordinates found
                for (ex,ey,ew,eh) in eyes:
                        # develop the roi for eyes (region of interest)
                        roi_eyes = faceLocate2[ey: ey + eh, ex: ex + ew]
                        # resize the glasses according to the eye coordinates
                        glasses2 = cv2.resize(glasses.copy(),(ew,eh))

                        # define glasswidth, glassheight
                        gw, gh, gc = glasses2.shape
                        # loop through gw and gh
                        # until the alpha chanel is 0
                        # this clears the background
                        for i in range(0,gw):
                                for j in range(0,gh):
                                        if glasses2[i,j][2] != 0:
                                                # assign the right coordinates from face to the glasses filter
                                                faceLocate[ey+i,ex+j] = glasses2[i,j]
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

video_capture.release()
cv2.destroyAllWindows()

                


