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
nose_cascade = cv2.CascadeClassifier(r"C:\Users\asras\Desktop\uni\So-S1\Python\Project\data\Nariz.xml")
mouth_cascade = cv2.CascadeClassifier(r"C:\Users\asras\Desktop\uni\So-S1\Python\Project\data\Mouth.xml")

class filterWindow:
        def __init__(self,wnd):
                self.frame = Tkinter.Frame(wnd,height = 100, width = 200)
                self.frame.pack()

                self.f1Btn = Tkinter.Button(wnd, text = "Sun Glasses", command = self.placef1)
                self.f1Btn.pack()

                self.f2Btn = Tkinter.Button(wnd, text = "Mustache", command = self.placef2)
                self.f2Btn.pack()

                self.ExBtn = Tkinter.Button(wnd, text="Exit", command = wnd.destroy)
                self.ExBtn.pack()

                self.photo = cv2.imread('s3.jpg')
                self.photo2 = cv2.cvtColor(self.photo,cv2.COLOR_BGR2RGB)

                self.photo3 = ImageTk.PhotoImage(image = Image.fromarray(self.photo2))
                self.panelA = Tkinter.Label(wnd, width = 800,height = 400)
                self.panelA.pack(padx = 10, pady = 10)

                self.panelA.configure(image = self.photo3)
                self.panelA.image = self.photo2

                self.faces = face_cascade.detectMultiScale(self.photo,1.3,5)


        def placef1(self):
                glasses = cv2.imread('sunglasses.png')
                photo = self.photo
                #photo1 = self.photo1
                gray = cv2.cvtColor(photo,cv2.COLOR_BGR2GRAY)
                #faces = face_cascade.detectMultiScale(photo,1.3,5)
                # loop through the face found
                for (x,y,w,h) in self.faces:
                        # store face coordinates, width and height
                        faceBox = (x,y,w,h)
                        # store a box with just the face
                        # one stored in grayscale and one in normal form
                        faceLocate = photo[y:y+h, x:x+w]
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

                img = cv2.cvtColor(photo,cv2.COLOR_BGR2RGB)
                photo1 = ImageTk.PhotoImage(image = Image.fromarray(img))
                self.panelA.configure(image = photo1)
                self.panelA.image = photo1

                        
        def placef2(self):
                mos = cv2.imread('mustache.png')
                photo = self.photo
                gray = cv2.cvtColor(photo,cv2.COLOR_BGR2GRAY)
                #faces = face_cascade.detectMultiScale(gray,1.3,5)

                # loop through the face found
                for (x,y,w,h) in self.faces:
                        # store face coordinates, width and height
                        faceBox = (x,y,w,h)
                        # store a box with just the face
                        # one stored in grayscale and one in normal form
                        faceLocate = photo[y:y+h, x:x+w]
                        faceLocate2 = gray[y:y+h, x:x+w]
                        
                        nose = nose_cascade.detectMultiScale(faceLocate2)
                        # loop through nose coordinates
                        # if only one nose found
                        # this reduces error
                        if len(nose) == 1:
                                for (nx,ny,nw,nh) in nose:
                                        #print nx, ny, nw, nh
                                        #print nose
                                        #print len(nose)
                                        # find the region of interest on the face
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
                                        

                img = cv2.cvtColor(photo,cv2.COLOR_BGR2RGB)
                photo1 = ImageTk.PhotoImage(image = Image.fromarray(img))
                self.panelA.configure(image = photo1)
                self.panelA.image = photo1

                        

wnd = Tkinter.Tk()
wnd.title("My Filters")
app = filterWindow(wnd)
wnd.mainloop()


