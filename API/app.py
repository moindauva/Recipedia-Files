import cv2
import numpy as np
from flask import Flask, request,  jsonify, make_response
import base64

app = Flask(__name__)


def readBase64(uri):
    try:
        encoded_data = uri.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except:
        return None


@app.route('/api/detect', methods=['POST'])
def detect():
    try:
        data = request.get_json()
        final_output = []

        if 'img' in data:

            net = cv2.dnn.readNet("yolov3_custom_train_final.weights",
                                  "yolo.cfg")

            classes = ["Cabbage", "Potato", "Tomato",
                       "Apple", "Banana", "Grapes"]

            layer_names = net.getLayerNames()
            outputlayers = [layer_names[i-1]
                            for i in net.getUnconnectedOutLayers()]

            img = readBase64(data['img'])
            img = cv2.resize(img, None, fx=0.3, fy=0.3)
            height, width, channels = img.shape

            blob = cv2.dnn.blobFromImage(
                img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

            net.setInput(blob)
            outs = net.forward(outputlayers)

            class_ids = []
            confidences = []
            boxes = []

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
                        class_ids.append(class_id)

                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                font = cv2.FONT_HERSHEY_PLAIN
                for i in range(len(boxes)):
                    if i in indexes:
                        x, y, w, h = boxes[i]
                        label = str(classes[class_ids[i]])
                        if(label not in final_output):
                            final_output.append(str(classes[class_ids[i]]))

        return make_response(jsonify(data=final_output),200)
    except:
        return make_response(jsonify(data=[]), 400)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
