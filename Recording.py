from flask import Flask, Response, render_template
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import json

app = Flask(__name__)

# Load the four YOLO models
model1 = YOLO(r'C:\Users\simo2\source\repos\FassehTest\FassehTest\Confidence_Score.pt')
model2 = YOLO(r'C:\Users\simo2\source\repos\FassehTest\FassehTest\Eyes_Gaze.pt')
model3 = YOLO(r'C:\Users\simo2\source\repos\FassehTest\FassehTest\Face_Expression.pt')
model4 = YOLO(r'C:\Users\simo2\source\repos\FassehTest\FassehTest\Stand.pt')

def get_results(frame,frameID):
  results1 = model1(frame)
  results2 = model2(frame)
  results3 = model3(frame)
  results4 = model4(frame)

  image1 = Image.fromarray(np.array(results1[0].plot())).convert('L')
  image2 = Image.fromarray(np.array(results2[0].plot())).convert('L')
  image3 = Image.fromarray(np.array(results3[0].plot())).convert('L')
  image4 = Image.fromarray(np.array(results4[0].plot())).convert('L')

  combined_image = Image.new('RGB', (image1.width, image1.height))
  if(frameID==0):
    combined_image.paste(image1)
  elif(frameID==1):
    combined_image.paste(image2)
  elif(frameID==2):
    combined_image.paste(image3)
  elif(frameID==3):
    combined_image.paste(image4)

  return np.array(combined_image)


def generate_frames():
  frameID=-1
  camera = cv2.VideoCapture(0)  # Use 0 for default webcam
  while True:
    success, frame = camera.read()
    if(frameID<4):
       frameID+=1
    else:
       frameID=0
    if not success:
      break
    else:
      result_image = get_results(frame,frameID)
      ret, buffer = cv2.imencode('.jpg', result_image)
      frame = buffer.tobytes()
      yield (b'--frame\r\n'
             b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
      
@app.route('/')
def index():
    return render_template('results.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)




