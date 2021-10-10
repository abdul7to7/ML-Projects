# NOTE : Run this python file in Command Prompt/Terminal
# USAGE - Go to the directory where the source files are kept via Command Prompt/Terminal. Then run the following command.
# python scan.py 
# Don't Forget to change the path of file to open the document at line 145.

# import the necessary packages

import numpy as np
import cv2
import face_recognition
import imutils
import pickle
from datetime import datetime
import time
import os

import easyocr
from PIL import Image,ImageDraw

import tkinter
from tkinter import filedialog

def Faces(imgtoprocess):


    #find path of xml file containing haarcascade file
    cascPathface = os.path.dirname(
    cv2.__file__) + "\\data\\haarcascade_frontalface_alt2.xml"
    # load the harcaascade in the cascade classifier
    faceCascade = cv2.CascadeClassifier(cascPathface)
    # load the known faces and embeddings saved in last file
    data = pickle.loads(open('face_enc', "rb").read())
    #Find path to the image you want to detect face and pass it here
    image = imgtoprocess
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #convert image to Greyscale for haarcascade
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                     scaleFactor=1.1,
                                     minNeighbors=5,
                                     minSize=(60, 60),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
 
    # the facial embeddings for face in input
    encodings = face_recognition.face_encodings(rgb)
    names = []
    # loop over the facial embeddings incase
    # we have multiple embeddings for multiple fcaes
    for encoding in encodings:
        #Compare encodings with encodings in data["encodings"]
        #Matches contain array with boolean values and True for the embeddings it matches closely
        #and False for rest
        matches = face_recognition.compare_faces(data["encodings"],
        encoding)
        #set name =inknown if no encoding matches
        name = "Unknown"
        # check to see if we have found a match
        if True in matches:
            #Find positions at which we get True and store them
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                #Check the names at respective indexes we stored in matchedIdxs
                name = data["names"][i]
                #increase count for the name we got
                counts[name] = counts.get(name, 0) + 1
                #set name which has highest count
                name = max(counts, key=counts.get)

 
        # update the list of names
            names.append(name)
    return names        


def Objs(imagetoprocess):
    yolo = cv2.dnn.readNet("E:\\Env\\ClassProject\\ObjectDetection\\yolov3.weights", "E:\\Env\\ClassProject\\ObjectDetection\\yolov3.cfg")
    classes = []

    with open("E:\\Env\\ClassProject\\ObjectDetection\\coco.names", "r") as file:
        classes = [line.strip() for line in file.readlines()]
    layer_names = yolo.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in yolo.getUnconnectedOutLayers()]

    colorRed = (0,0,255)
    colorGreen = (0,255,0)

    # #Loading Images
    img = imagetoprocess
    height, width, channels = img.shape

    # # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    yolo.setInput(blob)
    outputs = yolo.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    labels=[]
    for i in range(len(boxes)):
        if i in indexes:
            labels.append(str(classes[class_ids[i]]))    
    return labels


def Texts(imagetoprocess):
    reader=easyocr.Reader(['en'],gpu=False)

    bounds=reader.readtext(imagetoprocess)
    l=[]
    for i in bounds:
        l.append(i[1])
    return l        
 
main_win = tkinter.Tk() 
main_win.withdraw()

main_win.overrideredirect(True)
main_win.geometry('0x0+0+0')

main_win.deiconify()
main_win.lift()
main_win.focus_force()
 
main_win.sourceFile = filedialog.askopenfilename(filetypes = (("Image Files",("*.jpg","*.png","*.jpeg")),("All Files","*")),parent=main_win, initialdir= "/",
title='Please select a image file')

main_win.destroy()

img_path = main_win.sourceFile

image = cv2.imread(img_path)
f=Faces(image)
o=Objs(image)
t=Texts(image)
sf=''
so=''
st=''
for i in f:
    sf=sf+i+' '
for j in o:
    so=so+j+' '
for k in t:
    st=st+k+' '    


file=open('output.txt','a')

file.write('Faces: ')
file.writelines(sf)
file.write('\n')
file.write('Objects: ')
file.write(so)   
file.write('\n')
file.write('Chars: ')
file.write(st)
file.write('\n\n')
file.close()



