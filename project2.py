# Name: Asra Saeed
# AndrewID: asras
# Project: filters

import cv2

# For face detection and detecting other facial features,
# I used openCv's Haar cascades
# So far, I used a face cascade, a eye cascade and nose cascade
# for implementing more filters, I might use more

#__________INITIALIZATION________________________

# load cascade to use
# I had to download the xml files on my computer,
# and link them to cascadeclassifiers to load them
face_cascade = cv2.CascadeClassifier(r"C:\Users\asras\Desktop\uni\So-S1\Python\Project\data\haarcascades\haarcascade_frontalface_alt.xml")
eye_cascade = cv2.CascadeClassifier(r"C:\Users\asras\Desktop\uni\So-S1\Python\Project\data\frontalEyes35x16.xml")
nose_cascade = cv2.CascadeClassifier(r"C:\Users\asras\Desktop\uni\So-S1\Python\Project\data\Nariz.xml")

# load pictures to use
# the first picture can be changed to other portraits,
# to display filters on specific faces
pic = cv2.imread('s1.jpg')

# then I loaded the filters that I will be using
# first I implemented the glasses filter
glasses = cv2.imread('sunglasses.png')
#secondly I implemented the mustache filter
mos = cv2.imread('mustache.png')

#___________MAIN CODE_____________________________

# Firstly, converted the picture to grayscale
grayPic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)

#____initial steps for mustache filter___

# convert filter to grayscale
grayMos = cv2.cvtColor(mos, cv2.COLOR_BGR2GRAY)

# define mask and mask_inv
# masks are created in order to get rid of the background
# here we set a threshold which converts all pixels above 220 to 225,
# and below 220 to 0 (black)
# white pixels are considered background, we invert the mask,
# to work with pixels we want
rec, orig_mask = cv2.threshold(grayMos, 220,255,cv2.THRESH_BINARY_INV)
orig_mask_inv = cv2.bitwise_not(orig_mask)

# original image size for mustache is saved
# to use it for re sizing purposes
origmosHeight, origmosWidth = mos.shape[:2]

#____________________________________________

# detect face in picture using the cascade
faces = face_cascade.detectMultiScale(grayPic,1.3,5)

# loop through the face found
for (x,y,w,h) in faces:
        # store face coordinates, width and height
        faceBox = (x,y,w,h)
        # store a box with just the face
        # one stored in grayscale and one in normal form
        faceLocate = pic[y:y+h, x:x+w]
        faceLocate2 = grayPic[y:y+h, x:x+w]
        
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
        # Mustache
        # repeat the same process as glasses
        # find the nose from the cascade
        nose = nose_cascade.detectMultiScale(faceLocate2)
        # loop through nose coordinates
        for (nx,ny,nw,nh) in nose:
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
                                        faceLocate[ny+int(nh/2.0)+i, nx+j] = mos2[i,j]


# use imshow to display the image
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.imshow('image',pic)
cv2.waitKey(0)
cv2.destroyAllWindows()



