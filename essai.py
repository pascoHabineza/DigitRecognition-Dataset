import os
import cv2
import glob
import time
import mahotas
from dataset import HOG
from skimage.feature import hog
import numpy as np
from sklearn.externals import joblib
model= joblib.load("greatness.pkl")
# hog= HOG(orientations= 9, pixelsPerCell=(14,14), cellsPerBlock=(1,1))
indir= "This PC/Downloads/digit_images_p1"
indir= "C:\\Users\\clinic18\\Desktop\\testdata"
contours=[]
for root, dirs, filenames in os.walk(indir):
    for f in filenames:
        
        full_filename = indir + "\\" + f
        image= cv2.imread(full_filename)
        img= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image= cv2.GaussianBlur(img, (5, 5), 0)
        image = cv2.Canny(image, 30, 150)
        im2, cnts, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        print(cnts)
        print(len(cnts))
        print("f is", f)
        # time.sleep(1)
        #import the time so that you can slow the the process down to see if the thing is printing one digit at time
        # countours= contours+cnts\
        for i in cnts:
            (x,y, w, h)= cv2.boundingRect(i)
            if w>=3 and h>=9:
                roi = img[y:y + h, x:x + w]
                thresh = roi.copy()
                T = mahotas.thresholding.otsu(roi)
                thresh[thresh > T] = 255
                thresh = cv2.bitwise_not(thresh)
                print(thresh.shape[0], thresh.shape[1])
                time.sleep(1)
                thresh= cv2.resize(thresh, (28,28), interpolation=cv2.INTER_AREA)
                # thresh= cv2.dilate(thresh,(0.1,0.1))

                

                hist= hog(thresh, orientations=9, pixels_per_cell=(14,14 ), cells_per_block=(1, 1))

                hist= np.array([hist], dtype="object")
                print(hist)
                # time.sleep(0.5)
                digit=model.predict(hist)

                print("this number is", digit)
                time.sleep(1.5)
                with open("grateful.txt","a") as f:
                    print(digit, file=f, end=" ")
                cv2.imshow("output", thresh)
                cv2.waitKey(100)


# digits= [ cv2.imread(file) for file n glob.glob("This PC/Downloads/digit_images_p1/*.PNG")]
# for i in digits:
#     cv
        