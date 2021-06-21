import FaceDetection as FD
import os.path
from os import listdir
from os.path import isfile, join

if __name__ == '__main__':
    mypath = "ImageTest/"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    print(onlyfiles)
    for f in onlyfiles:
        path = mypath + f
        FD.face_detection(path, f)
