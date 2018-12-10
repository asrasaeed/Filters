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
nose_cascade = cv2.CascadeClassifier(r"C:\Users\asras\Desktop\uni\So-S1\Python\Project\data\Nariz.xml")

video_capture = cv2.VideoCapture(0)

while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        mos = cv2.imread('mustache.png')

        for (x,y,w,h) in faces:
            # store face coordinates, width and height
            faceBox = (x,y,w,h)
            # store a box with just the face
            # one stored in grayscale and one in normal form
            faceLocate = frame[y:y+h, x:x+w]
            faceLocate2 = gray[y:y+h, x:x+w]
            nose = nose_cascade.detectMultiScale(faceLocate2)
            # loop through nose coordinates
            if len(nose) == 1:
                for (nx,ny,nw,nh) in nose:
                    roi_nose = faceLocate2[ny:ny+nh, nx: nx + nw]
                    # resize the filter accordingly
                    mos2 = cv2.resize(mos.copy(),(nw,nh))
                    # loop through the width and height of filter
                    mw, mh, mc = mos2.shape
                    for i in range(0, mw):
                        for j in range(0, mh):
                            if mos2[i,j][2] != 0:
                                # assign the coordinates to the face
                                faceLocate[ny+i, nx+j] = mos2[i,j]
                    else:
                        (nx,ny,nw,nh) = nose[0]
                        #find the region of interest on the face
                        roi_nose = faceLocate2[ny:ny+nh, nx: nx + nw]
                        # resize the filter accordingly
                        mos2 = cv2.resize(mos.copy(),(nw,nh))
                        # loop through the width and height of filter
                        mw, mh, mc = mos2.shape
                        for i in range(0, mw):
                            for j in range(0, mh):
                                if mos2[i,j][2] != 0:
                                    # assign the coordinates to the face
                                    faceLocate[ny+i, nx+j] = mos2[i,j]
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
video_capture.release()
cv2.destroyAllWindows()
