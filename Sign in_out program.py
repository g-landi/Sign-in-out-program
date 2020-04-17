import numpy as np
import cv2
import math
import os
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import time
import pickle
import datetime as dt
from datetime import datetime
import keyboard



def findlocation(ListL):

    A = 0
    B = 0
    C = 0
    D = 0
    loc = "temploc"

    for x in ListL:
        if (x == "A"):
            A+=1
        if (x == "B"):
            B+=1
        if (x == "C"):
            C+=1
        if (x == "D"):
            D+=1

    if A > max(B, C, D):
        loc = "Bathroom"
    elif B > max(A, C, D):
        loc = "SRC"
    elif C > max(A, B, D):
        loc = "Other Teacher"
    elif D > max(A, C, B):
        loc = "Office"

    return loc


def findrotation(image_1):

        orb = cv2.ORB_create()
        cap = cv2.VideoCapture(1)
        image_1 = resize(image_1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
        goodmatch = True
        time = True
        ListL = []
        count  = 0

        for lp in range(4):
            ret, frame = cap.read()  # return a single frame in variable `frame
            cv2.imwrite(r"C:\Users\gaell\OneDrive\Documents\SCE- Schepens\Official project\Students\StudentFound.jpg", frame)


        while goodmatch:
            count+=1
            ret,frame = cap.read()

            image_2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            kp1, des1 = orb.detectAndCompute(image_1, None)
            kp2, des2 = orb.detectAndCompute(image_2, None)

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            matches = bf.match(des1, des2)
            good = []

            for m in matches:
                if m.distance < 30:
                    good.append(m)

            #matches = sorted(matches, key=lambda x: x.distance)
            #good = matches[:15]

            # print(good)

            img1_p = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            img2_p = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(img1_p, img2_p, cv2.RANSAC, 5.0)

            # print(M)

            theta = - math.atan2(M[0, 1], M[0, 0]) * 180 / math.pi

            destination(theta, ListL)

            print(theta)

            img3 = cv2.drawMatches(image_1, kp1, image_2, kp2, good, None, flags=2)
            cv2.imshow("output", img3)
            cv2.waitKey(1)
            if (count == 70):
                goodmatch = False

        currentDT = datetime.now()
        ti = (currentDT.strftime("%I:%M:%S %p"))
        print(ti)
        datee = (currentDT.strftime("%a, %b %d, %Y"))
        Final = findlocation(ListL)

        while(time):
            if keyboard.is_pressed('b'):
                currentT = datetime.now()
                to = (currentT.strftime("%I:%M:%S %p"))
                print (to)
                time = False


        return datee, ti, to, Final


def SignInOut(student,datee,ti,to,location):
    sign = open(r"C:\Users\gaell\OneDrive\Documents\SCE- Schepens\Official project\Sign in out sheet.txt", "a")

    sign.write("/n" "Name: {0} \n" "Date: {1} \n" "Time out: {2} \n" "Time in: {3} \n" "Location: {4}\n".format(student,datee,ti,to,location) )




def whichstudent():
    pickle_in = open(r"\Users\gaell\OneDrive\Documents\SCE- Schepens\Official project\TrianData\X", "rb")
    X = pickle.load(pickle_in)

    pickle_in = open(r"\Users\gaell\OneDrive\Documents\SCE- Schepens\Official project\TrianData\Y", "rb")
    y = pickle.load(pickle_in)

    CATEGORIES = ["Student1", "Student2", "Student3", "Student4", "Student5"]

    def prepare(filepath):
        IMG_SIZE = 50  # 50 in txt-based
        img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # read in the image, convert to grayscale
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
        return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # return the image with shaping that TF wants.

    model = tf.keras.models.load_model(
        r"C:\Users\gaell\OneDrive\Documents\SCE- Schepens\Official project\TrianData\64x3-CNN.model")

    prediction = model.predict(
        [prepare(r"C:\Users\gaell\OneDrive\Documents\SCE- Schepens\Official project\Students\StudentFound.jpg")])
    return(CATEGORIES[int(prediction[0][0])])

#def AddStudent():

def AddStudentt():

    cap = cv2.VideoCapture(1)  # video capture source camera (Here webcam of laptop)

    while True:
        ret, frame = cap.read()  # return a single frame in variable `frame
        cv2.imshow('img1', frame)  # display the captured image
        if cv2.waitKey(1) & 0xFF == ord('s'):  # save on pressing 's'
            ret, frame = cap.read()  # return a single frame in variable `frame
            cv2.imwrite(r"C:\Users\gaell\OneDrive\Documents\SCE- Schepens\Official project\Students\Student1.jpg",frame)


            break

    cap.release()
    cv2.destroyAllWindows()

def resize(image):

    #scale_percent = 10  # percent of original size
    width =  500                               #int(image.shape[1] * scale_percent / 100)
    height =  500                              #int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA) # resize image
    return resized

def destination(degree, ListL):

    degree = int(degree)

    if (-45 <= degree <= 45):
        location = "A"
        ListL.append(location)

        #print ("Correct:" + str(degree))

    if (45 < degree <= 135):
        location = "D"
        ListL.append(location)
        #print("D: Office")
        #print("Correct:" + str(degree))

    if (135 < degree <= 180 or -180 < degree <= -135  ):
        location = "C"
        ListL.append(location)
        #print("C: Bathroom")
        #print ("Correct:" + str(degree))

    if ( -135 < degree <= -45 ):
        location = "B"
        ListL.append(location)
        #print("B: Teacher")
        #print("Correct:" + str(degree))




for lp in range(1000):
    Student = whichstudent()

    if (Student == "Student1"):
        image_1 = cv2.imread(r'C:\Users\gaell\OneDrive\Documents\SCE- Schepens\Official project\Students\Student1.jpg',cv2.IMREAD_GRAYSCALE)
        s = "Student1"

    if (Student == "Student2"):
         image_1 = cv2.imread(r'C:\Users\gaell\OneDrive\Documents\SCE- Schepens\Official project\Students\Student2.jpg',
         cv2.IMREAD_GRAYSCALE)
         s = "Student2"
    if (Student == "Student3"):
         image_1 = cv2.imread(r'C:\Users\gaell\OneDrive\Documents\SCE- Schepens\Official project\Students\Student3.jpg',
         cv2.IMREAD_GRAYSCALE)
         s = "Student3"
    if (Student == "Student4"):
         image_1 = cv2.imread(r'C:\Users\gaell\OneDrive\Documents\SCE- Schepens\Official project\Students\Student4.jpg',
         cv2.IMREAD_GRAYSCALE)
         s = "Student4"
    if (Student == "Student5"):
         image_1 = cv2.imread(r'C:\Users\gaell\OneDrive\Documents\SCE- Schepens\Official project\Students\Student5.jpg',
         cv2.IMREAD_GRAYSCALE)
         s = "Student5"

    d,ti,to,l = findrotation(image_1)
    SignInOut(s,d,ti,to,l)




