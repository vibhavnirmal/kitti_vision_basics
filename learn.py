import os
import numpy as np
import cv2


def read_images(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

imgs = read_images('dataset37\image_01\data')


# monoslam 

for i in range(len(imgs)):
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(imgs[i],None)

    img = cv2.drawKeypoints(imgs[i],kp,None,color=(0,255,0), flags=0)

    cv2.imshow('image',img)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break