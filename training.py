# USAGE
# python train.py --dataset data/digits.csv --model models/svm.cpickle

# import the necessary packages
import exp
from sklearn.svm import LinearSVC
from dataset import HOG
import dataset
import argparse
# mport _Pickle as cPicklei
from sklearn.externals import joblib
import numpy as np
from skimage.feature import hog


(digits, target) = dataset.loadDigits()
data = []
size=(28,28)
# initialize the HOG descriptor
# hog = HOG(orientations = 9, pixelsPerCell = (14, 14),
# 	cellsPerBlock = (1, 1))

# loop over the images
for image in digits:
	
	hist = hog(image.reshape((28,28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), transform_sqrt=True)

	# hist = hog.describe(image)
	data.append(hist)
hog_features= np.array(data, dtype="object")
# train the model
model = LinearSVC(random_state = 42)
model.fit(hog_features, target)

# dump the model to file
# f = open(MODEL, "w")
# f.write(cPickle.dumps(model))
# f.close()
joblib.dump(model,"animalFarm.pkl", compress=3)
#see how to use the HOG whatever now that the MNIST dataset is working properly