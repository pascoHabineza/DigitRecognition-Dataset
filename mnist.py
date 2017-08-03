from sklearn.externals import joblib
model= joblib.load("digits_cls.pkl")
import cv2
import time
import os
import sklearn.datasets
import numpy as np
import time
import mahotas
import exp 
# hog= HOG(orientations= 9, pixelsPerCell=(14,14), cellsPerBlock=(1,1))
def donne():
    # indir= "This PC/Downloads/digit_images_p1"
    indir= "C:\\Users\\clinic18\\Desktop\\tdata"
    contours=[]
    data=[]
    List=[]
    e=0
    target=[]
    for root, dirs, filenames in os.walk(indir):
        for f in filenames:
            
            full_filename = indir + "\\" + f
            # print(full_filename)
          
            image= cv2.imread(full_filename)
            # cv2.imshow("targetImage",image)
            # cv2.waitKey(100)
            # # time.sleep(0.4)
            
            # print(image.shape[0], image.shape[1])
         
            #image= cv2.resize(image,(28,28),interpolation=cv2.INTER_AREA)
            img= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            blurred= cv2.GaussianBlur(img, (5, 5), 0)
            
            thresh=img.copy()
            T = mahotas.thresholding.otsu(blurred)
            thresh[thresh > T] = 255
            thresh = cv2.bitwise_not(thresh)
            e=e+1
            # print(roi.shape)
            # print(thresh.shape[0],thresh.shape[1],e, f)
            cv2.imshow("test",thresh)
            cv2.waitKey(100)
            # time.sleep(0.5)
            thresh= exp.center_extent(thresh, (28,28))
            # print(thresh.shape)
            # print(thresh)
            for i in range(thresh.shape[0]):
                for e in range(thresh.shape[1]):
                    pixel=thresh[i, e]
                    pixel= int(pixel)
                    List+=[pixel]

            # print(len(List))
            # time.sleep(0.5)
            data+=[List]
            # print(data)
            # time.sleep(1.5)
            List=[]


    target= [0, 5, 0, 0, 6, 0, 6, 9, 7, 7, 6, 2, 1, 3, 4, 6, 2, 2, 1, 3, 3, 0, 8, 8, 2, 9, 8, 9, 8, 9, 3, 8, 5, 2, 6, 9, 6, 9, 2, 6, 5, 4, 9, 9, 5, 5, 7, 7, 6, 9, 7, 6, 5, 7, 6, 4, 4, 3, 4, 4, 5, 3, 4, 3, 2, 6, 5, 4, 4, 7, 4, 3, 7, 7, 5, 3, 6, 4, 6, 3, 8, 3, 3, 7, 0, 4, 7, 5, 5, 4, 3, 3, 8, 2, 0, 6, 8, 3, 9, 3, 6, 2, 8, 8, 1, 2, 8, 8, 9, 1, 7, 8, 0, 3, 3, 8, 7, 8, 3, 1, 8, 0, 0, 3, 0, 0, 1, 1, 1, 9, 3, 5, 2, 9, 2, 0, 7, 5, 1, 8, 8, 8, 1, 9, 9, 9, 1, 1, 9, 7, 8, 4, 9, 8, 1, 2]
    data= np.array(data, dtype="uint8")
    target=np.array(target)
    dataset= sklearn.datasets.base.Bunch(data=data, target=target)
    print(type(dataset))
    return dataset 
    