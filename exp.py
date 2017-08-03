
import numpy as np
import cv2
# import imutils
image=cv2.imread("digit001.png")
import mahotas
size= (28,28)

def resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized
def center_extent(image, size):
    (eW, eH) = size
    print(image.shape[1],image.shape[0])
    if image.shape[1] > image.shape[0]:
        image = resize(image, width = eW)
    else:
        image =resize(image, height = eH)

    extent = np.zeros((eH, eW), dtype = "uint8")
    offsetX = (eW - image.shape[1]) / 2
    # print(offsetX)
    offsetY = (eH - image.shape[0]) / 2
    # print(offsetY)
    offsetX=int(offsetX)
    offsetY=int(offsetY)
    extent[offsetY:offsetY + image.shape[0], offsetX:offsetX + image.shape[1]] = image
    CM = mahotas.center_of_mass(extent)
    # print(CM)
    # print(type(CM))
    # print(len(CM))
    (cY, cX) = np.round(CM).astype("int16")
    # print(cY,cX)
    (dX, dY) = ((size[0] / 2) - cX, (size[1] / 2) - cY)
    M = np.float32([[1, 0, dX], [0, 1, dY]])
    extent = cv2.warpAffine(extent, M, size)
    # print(extent.shape)
    return extent