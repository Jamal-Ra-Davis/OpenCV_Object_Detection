import cv2
import numpy as np
from pathlib import Path

#thres = 0.45 # Threshold to detect object

classNames = []
#classFile = "/home/pi/Desktop/Object_Detection_Files/coco.names"
classFile = 'C:\\Users\\Jamal\\Documents\\programming\\Python\\OpenCV\\Object_Detection_Files\\coco.names'
classFile_ = Path(classFile)


with open(classFile_,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

# configPath = "/home/pi/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
configPath = "C:\\Users\\Jamal\\Documents\\programming\\Python\\OpenCV\\Object_Detection_Files\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
configPath_ = Path(configPath)

# weightsPath = "/home/pi/Desktop/Object_Detection_Files/frozen_inference_graph.pb"
weightsPath = "C:\\Users\\Jamal\\Documents\\programming\\Python\\OpenCV\\Object_Detection_Files\\frozen_inference_graph.pb"
weightsPath_ = Path(weightsPath)


net = cv2.dnn_DetectionModel(weightsPath_, configPath_)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)





def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    #print(classIds,bbox)
    if len(objects) == 0: objects = classNames
    objectInfo =[]
    print(classNames)
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box,className])
                if (draw):
                    cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                    cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    return img,objectInfo


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3,771)
    cap.set(4,565)
    #cap.set(10,70)

    success, img = cap.read()
    if success:
        print("Captured image")
    else:
        print("Failed to capture image")
    squirrel_path = "C:\\Users\\Jamal\\Documents\\programming\\Python\\OpenCV\\Object_Detection_Files\\squirrel.bmp"
    squirrel_path_ = Path(squirrel_path)
    image = cv2.imread(squirrel_path_, cv2.IMREAD_UNCHANGED)
    

    print(image.shape)

    result, objectInfo = getObjects(image,0.45,0.2)
    print(result)
    print(objectInfo)
    
    cv2.imshow("Output",image)
    cv2.waitKey(1000)

    # Play video
    squirrel_video_path = "C:\\Users\\Jamal\\Documents\\programming\\Python\\OpenCV\\Object_Detection_Files\\squirrel_video.mp4"
    squirrel_video_path_ = Path(squirrel_video_path)
    cap = cv2.VideoCapture(squirrel_video_path_)

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error opening video file")

    # Read and display frames until the end of the video
    while cap.isOpened():
        ret, frame = cap.read()
        print(frame.shape)

        if not ret:
            break

        # Display the frame
        result, objectInfo = getObjects(frame,0.55,0.2)
        cv2.imshow('Frame', frame)

        # Wait for 25ms and check if the user pressed 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture object and close all windows
    cap.release()
    
    while False:
        success, img = cap.read()
        result, objectInfo = getObjects(img,0.45,0.2)
        #print(objectInfo)
        cv2.imshow("Output",img)
        cv2.waitKey(1)
