##https://www.youtube.com/watch?v=cdblJqEUDNo
##USAGE
## MobileNetSSDv2test.py -c -i..

import cv2
import time
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-c', '--confidence', type=float, metavar='', required=True, help='Nilai confidence yang digunakan',default=0.3)
parser.add_argument("-i", "--image", required=True, default="D:code\capstone-cameraunit\trials\pltsft.jpg",
    help="path to input image")
args = parser.parse_args()

treshold = args.confidence

# Load a model imported from Tensorflow
tensorflowNet = cv2.dnn.readNetFromTensorflow('ssd_mobilenet_v2_coco_2018_03_29.pb', 'ssd_mobilenet_v2_coco_2018_03_29.pbtxt')
 
# Input image
img = cv2.imread(args.image)
rows, cols, channels = img.shape
classes = open('coco.names').read().strip().split('\n')
print(classes)
# Use the given image as input, which needs to be blob(s).
tensorflowNet.setInput(cv2.dnn.blobFromImage(img, size=(320, 320), swapRB=True, crop=False))
 
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
    if score > treshold and int(detection[1])==1:
     
        left = detection[3] * cols
        top = detection[4] * rows
        right = detection[5] * cols
        bottom = detection[6] * rows
        box = [int(left),int(top),int(right-left),int(bottom-top)]
        boxes.append(box)
        confidences.append(float(round(score,3)))
        #draw a red rectangle around detected objects
        #cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)
        #cv2.putText(img,"confidence:"+str(round(score,3)),(int(left),int(top)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
        
indices = cv2.dnn.NMSBoxes(boxes,confidences,treshold,0.7)
print(len(confidences))
print("indices",indices)
print(len(indices))

def centroid(boxes, indices):
    center=np.zeros((len(indices),2),dtype=int)
    i=0
    for index in indices.flatten():
        x= boxes[index][0]+0.5*boxes[index][2]
        y= boxes[index][1]+0.5*boxes[index][3]
        center[i]=[int(x),int(y)]
        i+=1
    return center
centre=centroid(boxes,indices)
print(centre)
if len(indices) > 0:
        i1=0
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            print(w,h)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,255), 2)
            cv2.circle(img,tuple(centre[i1]),3,(0,255,0),2 )
            text = "Conf:"+str(round(confidences[i],3))
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (139,0,0), 1)
            i1+=1
# Show the image with a rectagle surrounding the detected objects 
cv2.imshow('Image', img)
cv2.waitKey()
cv2.destroyAllWindows()