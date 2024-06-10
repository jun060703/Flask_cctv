from flask import Flask, render_template, Response
import cv2
import numpy as np
app = Flask(__name__)

prototxt_path = "C:/programing_study/color_judge/flask/prototxt.txt"
model_path = "C:/programing_study/color_judge/flask/mobilenet_iter_73000.caffemodel"

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

classes = ["none", "person"]

webcam_video = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None
recording = False



def generate_frames():
    global webcam_video, net, classes, out, recording
    while True:
        ret, frame = webcam_video.read()
        if not ret:
            break

        (h, w) = frame.shape[:2]

        resized_frame = cv2.resize(frame, (300, 300))

        blob = cv2.dnn.blobFromImage(resized_frame, 0.007843, (300, 300), 127.5)

        net.setInput(blob)

        detections = net.forward()

        person_detected = False

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.9:
                class_id = int(detections[0, 0, i, 1])
                if class_id == 15: 
                    person_detected = True
                    class_name = classes[1] 

                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # 바운딩 박스를 더 작게 조정
                    scale = 0.5  # 박스 크기 조정 비율 (0.5이면 50% 크기로 축소)
                    centerX, centerY = (startX + endX) // 2, (startY + endY) // 2
                    new_width = int((endX - startX) * scale)
                    new_height = int((endY - startY) * scale)
                    new_startX = max(centerX - new_width // 2, 0)
                    new_startY = max(centerY - new_height // 2, 0)
                    new_endX = min(centerX + new_width // 2, w)
                    new_endY = min(centerY + new_height // 2, h)

                    color = (0, 255, 0)
                    cv2.rectangle(frame, (new_startX, new_startY), (new_endX, new_endY), color, 2)
                    y = new_startY - 15 if new_startY - 15 > 15 else new_startY + 15
                    cv2.putText(frame, class_name, (new_startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    ###########################################################
        if person_detected and not recording:
            out = cv2.VideoWriter('static/사람감지!!.avi', fourcc, 20.0, (w, h))
            recording = True
        
        if recording:
            out.write(frame)  

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recorded')
def index():
    return render_template('recorded.html')

@app.route('/time')
def index():
    return render_template('time.html')        

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0', port=9900)
