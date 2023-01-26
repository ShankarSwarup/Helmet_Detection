from flask import Flask,render_template
from flask_wtf import FlaskForm
from wtforms import FileField , SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired
import cv2
import numpy as np
import imutils
from tensorflow.keras.models import load_model

app =  Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

net = cv2.dnn.readNet("yolov3-custom_7000.weights", "yolov3-custom.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


model = load_model('helmet-nonhelmet_cnn.h5')
print('model loaded!!!')


class UploadFileForm(FlaskForm):
    file = FileField("File",validators=[InputRequired()])
    submit = SubmitField("Upload File")



def helmet_or_nohelmet(helmet_roi):
	try:
		helmet_roi = cv2.resize(helmet_roi, (224, 224))
		helmet_roi = np.array(helmet_roi,dtype='float32')
		helmet_roi = helmet_roi.reshape(1, 224, 224, 3)
		helmet_roi = helmet_roi/255.0
		return int(model.predict(helmet_roi)[0][0])
	except:
			pass

@app.route('/',methods = ["GET","POST"])
@app.route('/home',methods = ["GET","POST"])
def home():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        directory = file.filename
        seperator = '.'
        directory = directory.split(seperator,1)[0]
        parent = os.getcwd().replace("\\","/") + '/static/numberplates'
        path = parent+'/'+directory
        if not os.path.exists(path):
            os.mkdir(path)
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename)))

        cap = cv2.VideoCapture(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename)))
        # list_frame  = []

        # len_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        COLORS = [(0,255,0),(0,0,255)]
        xii1 = 0

        layer_names = net.getLayerNames()
        output_layers = net.getUnconnectedOutLayersNames()
        

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter('output.avi', fourcc, 5,(888,500))

        ret = True
        xii = 1

        while ret:

            ret, img = cap.read()
            if img is None or xii1==200:
                break
            # list_frame.append(frame)
            # if len_frames == len(list_frame):
            #     break
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
                        # cv2.rectangle(img, (x, y), (x + w, y + h), color, 7)
                        if y_h>0 and x_h>0:
                            h_r = img[y_h:y_h+h_h , x_h:x_h +w_h]
                            c = helmet_or_nohelmet(h_r)
                            if c==1:
                                imagg = img[y:y+h,x:x+w]
                                cv2.imwrite('{}_{}.{}'.format(os.path.join(path,'img'), str(xii), 'jpg'), imagg)
                                xii = xii + 1
                            # cv2.putText(img,['helmet','no-helmet'][c],(x,y-100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)                
                            # cv2.rectangle(img, (x_h, y_h), (x_h + w_h, y_h + h_h),(255,0,0), 10)


            writer.write(img)
            xii1 = xii1+1
            # cv2.imshow("Image", img)

            if cv2.waitKey(1) == 27:
                break

        writer.release()
        cap.release()
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        image_names = os.listdir(path)
        for i in range(len(image_names)):
            image_names[i] = '../static/numberplates/'+ directory+'/'+image_names[i]
        return render_template('images.html',image_name = image_names)
        # file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),path,secure_filename(file.filename)))
    return render_template('index.html',form = form)

if __name__ == '__main__':
    app.run(debug=True)