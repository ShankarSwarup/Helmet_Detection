# import cv2
# import numpy as np
# import os
# import imutils
# import pytesseract
# from matplotlib import pyplot as plt
# from tensorflow.keras.models import load_model
# # import easyocr
# # from paddleocr import PaddleOCR
# pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"


# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# net = cv2.dnn.readNet("yolov3-custom_7000.weights", "yolov3-custom.cfg")
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


# model = load_model('helmet-nonhelmet_cnn.h5')
# print('model loaded!!!')

# cap = cv2.VideoCapture('video.mp4')
# COLORS = [(0,255,0),(0,0,255)]

# layer_names = net.getLayerNames()
# output_layers = net.getUnconnectedOutLayersNames()
 
# fourcc = cv2.VideoWriter_fourcc(*"XVID")
# writer = cv2.VideoWriter('output.avi', fourcc, 5,(888,500))

# final_text = ""

# def helmet_or_nohelmet(helmet_roi):
# 	try:
# 		helmet_roi = cv2.resize(helmet_roi, (224, 224))
# 		helmet_roi = np.array(helmet_roi,dtype='float32')
# 		helmet_roi = helmet_roi.reshape(1, 224, 224, 3)
# 		helmet_roi = helmet_roi/255.0
# 		return int(model.predict(helmet_roi)[0][0])
# 	except:
# 			pass

# ret = True

# while ret:

#     ret, img = cap.read()
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
 
#     im2 = img.copy()

#     # img = imutils.resize(img,height=500)
#     # img = cv2.imread('test.png')
#     height, width = img.shape[:2]

#     blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

#     net.setInput(blob)
#     outs = net.forward(output_layers)

#     confidences = []
#     boxes = []
#     classIds = []

#     for out in outs:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.3:
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)

#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)

#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 classIds.append(class_id)

#     indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

#     for i in range(len(boxes)):
#         if i in indexes:
#             x,y,w,h = boxes[i]
#             color = [int(c) for c in COLORS[classIds[i]]]
#             # green --> bike
#             # red --> number plate
#             if classIds[i]==0: #bike
#                 helmet_roi = img[max(0,y):max(0,y)+max(0,h)//4,max(0,x):max(0,x)+max(0,w)]
#             else: #number plate
#                 x_h = x-60
#                 y_h = y-350
#                 w_h = w+100
#                 h_h = h+100
#                 # cv2.rectangle(img, (x, y), (x + w, y + h), color, 7)

#                 if y_h>0 and x_h>0:
#                     cropped_image = img[ y:y+h , x:x+w ]
#                     h_r = img[y_h:y_h+h_h , x_h:x_h +w_h]
#                     if(cropped_image is not None):
#                         sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
#                         sharpen = cv2.filter2D(cropped_image, -1, sharpen_kernel)
#                         gray1 = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
#                         bfilter = cv2.bilateralFilter(gray1, 11, 17, 17)
#                         edged = cv2.Canny(bfilter,30,200)

#                         keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#                         contours = imutils.grab_contours(keypoints)
#                         contours = sorted(contours,key = cv2.contourArea , reverse = True)[:10]
                        
#                         location = []
#                         for contour in contours:
#                             approx = cv2.approxPolyDP(contour, 10, True)
#                             if len(approx) == 4:
#                                 location = approx
#                                 break
#                         # print(type(location))
                        
#                         if len(location)>0:
#                             mask = np.zeros(gray1.shape , np.uint8)
#                             new_image = cv2.drawContours(mask, [location], 0,255,-1)
#                             new_image = cv2.bitwise_and(cropped_image, cropped_image, mask = mask)
#                             (a , b ) = np.where(mask == 255)
#                             (a1 , b1) = (np.min(a) , np.min(b))
#                             (a2 , b2) = (np.max(a) , np.max(b))
#                             cropped_image1 = gray1[a1 : a2+1 , b1 : b2+1]                          # blur = cv2.GaussianBlur(gray1, (3,3), 0)
#                         #     # thresh = cv2.threshold(gray1, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

#                         #     # thresh = cv2.threshold(cropped_image1, 150, 255, cv2.THRESH_BINARY_INV)[1]

#                         #     # ocr = PaddleOCR(lang='en',rec_algorithm='CRNN')

#                         #     # result = ocr.ocr(cropped_image1, cls=False, det=False)

#                         #     # print(result)

#                             # img_text = reader.readtext(cropped_image1)
#                             # final_text = ""
#                             # for _, text, __ in img_text: # _ = bounding box, text = text and __ = confident level
#                             #         final_text += " "
#                             #         final_text += text
#                             # # print(final_text)

#                     cv2.namedWindow('finalImg', cv2.WINDOW_NORMAL)
#                     cv2.imshow("finalImg",cv2.cvtColor(cropped_image,cv2.COLOR_BGR2RGB))       
                        
#                     c = helmet_or_nohelmet(h_r)
#                     if(c == 1):
#                         print('No-helmet',final_text)
#                     cv2.putText(img,['helmet','no-helmet'][c],(x,y-100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)           
#                     # cv2.rectangle(img, (x_h, y_h), (x_h + w_h, y_h + h_h),(255,0,0), 10)


#     writer.write(img)
#     cv2.imshow("Image", img)

#     if cv2.waitKey(1) == 27:
#         break

# writer.release()
# cap.release()
# cv2.waitKey(0)
# cv2.destroyAllWindows()



import cv2
import numpy as np
import os
import imutils
from tensorflow.keras.models import load_model

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

net = cv2.dnn.readNet("yolov3-custom_7000.weights", "yolov3-custom.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


model = load_model('helmet-nonhelmet_cnn.h5')
print('model loaded!!!')

cap = cv2.VideoCapture('video.mp4')
COLORS = [(0,255,0),(0,0,255)]

layer_names = net.getLayerNames()
output_layers = net.getUnconnectedOutLayersNames()
 

fourcc = cv2.VideoWriter_fourcc(*"XVID")
writer = cv2.VideoWriter('output.avi', fourcc, 5,(888,500))


def helmet_or_nohelmet(helmet_roi):
	try:
		helmet_roi = cv2.resize(helmet_roi, (224, 224))
		helmet_roi = np.array(helmet_roi,dtype='float32')
		helmet_roi = helmet_roi.reshape(1, 224, 224, 3)
		helmet_roi = helmet_roi/255.0
		return int(model.predict(helmet_roi)[0][0])
	except:
			pass

ret = True

while ret:

    ret, img = cap.read()
    img = imutils.resize(img,height=500)
    # img = cv2.imread('test.png')
    height, width = img.shape[:2]

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    confidences = []
    boxes = []
    classIds = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)

                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                classIds.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            color = [int(c) for c in COLORS[classIds[i]]]
            if classIds[i]==0: #bike
                helmet_roi = img[max(0,y):max(0,y)+max(0,h)//4,max(0,x):max(0,x)+max(0,w)]
            else: #number plate
                x_h = x-60
                y_h = y-350
                w_h = w+100
                h_h = h+100
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 7)
                if y_h>0 and x_h>0:
                    h_r = img[y_h:y_h+h_h , x_h:x_h +w_h]
                    c = helmet_or_nohelmet(h_r)
                    cv2.putText(img,['helmet','no-helmet'][c],(x,y-100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)                
                    cv2.rectangle(img, (x_h, y_h), (x_h + w_h, y_h + h_h),(255,0,0), 10)


    writer.write(img)
    cv2.imshow("Image", img)

    if cv2.waitKey(1) == 27:
        break

writer.release()
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()