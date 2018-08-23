import cv2
import os

test_image_dir = "DatasetA_test/test"
imgListFile = open("DatasetA_test/image.txt", "r")
line = imgListFile.readline()
submit = open("submit.txt", "a")
while line:
    file_dir = os.path.join(test_image_dir, line)

   # print(file_dir)
    if not os.path.exists(file_dir):
        print("file_dir:",file_dir)
        print("%snot exists" % file_dir)
            #    return 0
    else:
        print("%s is exist" %file_dir)
    img = cv2.imread(os.path.join(test_image_dir, line))
   # print(os.path.join(test_image_dir, line))
    print(img.shape)
    line = imgListFile.readline()
"""
with open("DatasetA_test/image.txt", "r") as imgListFile:
    with open("submit.txt", "a") as submitFile:
        line = imgListFile.readline()
        while line:
            file_dir = os.path.join(test_image_dir, line)
            print(file_dir)
            if not os.path.exists(file_dir):
                print("image not exists")
            #    return 0
            img = cv2.imread(os.path.join(test_image_dir, line))
            print(os.path.join(test_image_dir, line))
            print(img.shape)
"""
#tasetA_test/image.txt", "r")if not os.path.exists("/home/lisren/Zeros/DatasetA_test/test/00ddbe75d7aff5037d360401af02ca57.jpg"):
#    print("yes")
#else:
#    print("exist")
#mg = cv2.imread("DatasetA_test/test/00ddbe75d7aff5037d360401af02ca57.jpg")
