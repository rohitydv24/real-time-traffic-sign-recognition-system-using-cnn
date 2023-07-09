import cv2
import numpy as np
import pandas as pd
import pyttsx3

# Load the trained model to classify signs
from tensorflow.keras.models import load_model

model = load_model('traffic_classifier.h5')

# Dictionary to label all traffic sign classes
classes = {1: 'Speed limit (20km/h)',
           2: 'Speed limit (30km/h)',
           3: 'Speed limit (50km/h)',
           4: 'Speed limit (60km/h)',
           5: 'Speed limit (70km/h)',
           6: 'Speed limit (80km/h)',
           7: 'End of speed limit (80km/h)',
           8: 'Speed limit (100km/h)',
           9: 'Speed limit (120km/h)',
           10: 'No passing',
           11: 'No passing veh over 3.5 tons',
           12: 'Right-of-way at intersection',
           13: 'Priority road',
           14: 'Yield',
           15: 'Stop',
           16: 'No vehicles',
           17: 'Veh > 3.5 tons prohibited',
           18: 'No entry',
           19: 'General caution',
           20: 'Dangerous curve left',
           21: 'Dangerous curve right',
           22: 'Double curve',
           23: 'Bumpy road',
           24: 'Slippery road',
           25: 'Road narrows on the right',
           26: 'Road work',
           27: 'Traffic signals',
           28: 'Pedestrians',
           29: 'Children crossing',
           30: 'Bicycles crossing',
           31: 'Beware of ice/snow',
           32: 'Wild animals crossing',
           33: 'End speed + passing limits',
           34: 'Turn right ahead',
           35: 'Turn left ahead',
           36: 'Ahead only',
           37: 'Go straight or right',
           38: 'Go straight or left',
           39: 'Keep right',
           40: 'Keep left',
           41: 'Roundabout mandatory',
           42: 'End of no passing',
           43: 'End no passing veh > 3.5 tons'
}

# Function to read the CSV file and get image paths and corresponding class IDs
def read_csv(csv_file):
    data = pd.read_csv(csv_file)
    paths = data['Path'].tolist()
    class_ids = data['ClassId'].tolist()
    return paths, class_ids

# Load the image and classify it
def classify(image):
    image = cv2.resize(image, (30, 30))
    image = np.expand_dims(image, axis=0)
    image = np.array(image) / 255.0
    pred = model.predict(image)[0]
    pred_class_index = np.argmax(pred)
    sign = classes[pred_class_index + 1]
    confidence = pred[pred_class_index] * 100
    print("Predicted Sign: ", sign)
    print("Confidence: {:.2f}%".format(confidence))
    return sign

# Create a video capture object
cap = cv2.VideoCapture(0)

# Initialize text-to-speech engine
engine = pyttsx3.init()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('Camera', frame)

    # Check for key press
    key = cv2.waitKey(1)

    # Capture an image when spacebar is pressed
    if key == ord(' '):
        # Classify the captured image
        predicted_sign = classify(frame)

        # Draw bounding box and class label on the captured frame
        pred_image = frame.copy()
        pred_image = cv2.resize(pred_image, (300, 300))
        cv2.rectangle(pred_image, (0, 0), (300, 40), (0, 0, 0), -1)
        cv2.putText(pred_image, "Predicted Sign: {}".format(predicted_sign), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow('Camera', pred_image)

        # Speak out the predicted sign using text-to-speech
        engine.say(predicted_sign)
        engine.runAndWait()

    # Break the loop when 'q' is pressed
    if key == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
