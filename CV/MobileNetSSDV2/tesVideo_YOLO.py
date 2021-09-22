import cv2
import numpy as np
import time
import argparse


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-c', '--confidence', type=float, metavar='', required=True, help='Nilai confidence yang digunakan',default=0.3)
args = parser.parse_args()

treshold = args.confidence

cap = cv2.VideoCapture("video_kelas.mp4")
#fourcc = cv2.VideoWriter
#out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
tensorflowNet = cv2.dnn.readNetFromTensorflow('ssd_mobilenet_v2_coco_2018_03_29.pb', 'ssd_mobilenet_v2_coco_2018_03_29.pbtxt')


while True:
    ret, frame = cap.read()
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rows, cols, channels = frame.shape
    classes = open('coco.names').read().strip().split('\n')
    print(classes)
    # Use the given image as input, which needs to be blob(s).
    tensorflowNet.setInput(cv2.dnn.blobFromImage(frame, size=(320, 320), swapRB=True, crop=False))
 
    # Runs a forward pass to compute the net output
    t0 = time.time()
    networkOutput = tensorflowNet.forward()
    t = time.time()
    print("time",t-t0)
    print(networkOutput[0,0,0,0])
    print(type(networkOutput))
    print(networkOutput.shape)
    print(networkOutput[0].shape)
    print("tesshape")
    print(networkOutput[0].shape)
    boxes=[]
    confidences=[]

    # Loop on the outputs
    for detection in networkOutput[0,0]:
    
        score = float(detection[2])
        if score > 0.3 and int(detection[1])==1:
     
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows
            box = [int(left),int(top),int(right-left),int(bottom-top)]
            boxes.append(box)
            confidences.append(float(round(score,3)))
 
            #draw a red rectangle around detected objects
            #cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)
            #cv2.putText(frame,"confidence:"+str(round(score,3)),(int(left),int(top)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
    
    indices = cv2.dnn.NMSBoxes(boxes,confidences,treshold,0.2)
    print(len(confidences))
    print(len(indices))
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            print(w,h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2)
            text = "Conf:"+str(round(confidences[i],3))
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (139,0,0), 1)
    
    cv2.imshow('frame',frame)
    #cv2.imshow('gray',gray)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.relase()
#out.relase()
cv2.destriyAllWindows()