from flask import Flask, redirect, url_for, render_template, Response
import cv2 as cv

app = Flask(__name__)

camera = cv.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            detector = cv.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
            eye_cascade = cv.CascadeClassifier('Haarcascades/haarcascade_eye.xml')
            faces = detector.detectMultiScale(frame, 1.3, 5) 
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)    

            for (x, y, w, h) in faces:
                cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2 )  




            ret, buffer = cv.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # we use yield instead of return because return will return only one frame but yield will return multiple frames
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/', methods=['GET'])
def welcome():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
