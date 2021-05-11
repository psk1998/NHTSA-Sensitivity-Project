def SSD_algo(img):
    import cv2
    ## Import SSD Algorithm
    weightsFile = 'D:/SP 21/Project/Algorithms/SSD/frozen_inference_graph.pb'
    cfgFile = 'D:/SP 21/Project/Algorithms/SSD/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    classFile = 'D:/SP 21/Project/Algorithms/SSD/coco.names'

    classNames = []
    with open(classFile,'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    net = cv2.dnn_DetectionModel(weightsFile,cfgFile)
    net.setInputSize(320,320)
    net.setInputScale(1.0/127.5)
    net.setInputMean((127.5,127.5,127.5))
    net.setInputSwapRB(True)

    def detection(image,time,windowName):
        classIds, confidence, bbox = net.detect(image,confThreshold=0.5)
        for classId, conf, box in zip(classIds.flatten(),confidence.flatten(),bbox):
            cv2.rectangle(image,box,color=(0,255,0))
            cv2.putText(image,classNames[classId-1],(box[0]+10,box[1]+10),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.4,(255,0,0))
        print(bbox.shape)
        return classIds, confidence, bbox

    classIds, confidence, bbox = detection(img,1000,'Algorithm detection')
    return classIds, confidence, bbox

def YOLO(img):
    import cv2
    import numpy as np
    ## Import YOLO Algorithm
    modelWeights = 'D:/SP 21/Project/Algorithms/YOLO/yolov3-tiny.weights'
    modelConfiguration = 'D:/SP 21/Project/Algorithms/YOLO/yolov3-tiny.cfg'
    classFile = 'D:/SP 21/Project/Algorithms/YOLO/coco.names'

    classNames = []
    with open(classFile,'rt') as f:
        classNames=f.read().rstrip('\n').split('\n')

    # Network Creation
    net = cv2.dnn.readNet(modelWeights,modelConfiguration)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img ,1/255 ,(416,416) ,(0,0,0) ,swapRB=True , crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    bbox = []
    confidences = []
    classIds = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if classId == 'person' and confidence > 0.75:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                
                # Convert into bottom left
                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                classIds.append(classId)

            elif classId != 'person' and confidence > 0.5:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                # Convert into bottom left
                x = int(center_x - w/2)
                y = int(center_y - h/2)

                bbox.append([x, y, w, h])
                confidences.append(float(confidence))
                classIds.append(classId)
    
    for classId, conf, box in zip(classIds,confidences,bbox):
            cv2.rectangle(img,box,color=(0,255,0))
            cv2.putText(img,classNames[classId-1],(box[0]+10,box[1]+10),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.4,(255,0,0))
    print(len(classIds))

    return classIds, confidence, bbox