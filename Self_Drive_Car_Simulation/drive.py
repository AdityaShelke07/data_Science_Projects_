import socketio
import eventlet
import numpy as np
from flask import Flask
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2
from tensorflow.keras.losses import MeanSquaredError

# Define and register a custom MSE function
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable()
def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Initialize SocketIO server
sio = socketio.Server()

# Flask app
app = Flask(__name__)
speed_limit = 10

# Preprocessing function for input images
def img_preprocess(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img

# Telemetry event handler
@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])
    steering_angle = float(model.predict(image))
    throttle = 1.0 - speed / speed_limit
    print('{} {} {}'.format(steering_angle, throttle, speed))
    send_control(steering_angle, throttle)

# Connect event handler
@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)

# Function to send control commands
def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })

# Main function
if __name__ == '__main__':
    # Load model with custom_objects argument to handle 'mse'
    model = load_model('model.h5', custom_objects={'mse': mse})
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
