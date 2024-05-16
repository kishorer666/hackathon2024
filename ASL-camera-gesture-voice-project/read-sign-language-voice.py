import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import pyttsx3
from picamera2 import Picamera2
import time
import cv2
import numpy as np

# Define data transformations
data_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 26)  # Assuming 26 classes for ASL A-Z

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the trained model
model = CNN()
model.load_state_dict(torch.load('asl_model.pth', map_location=torch.device('cpu')))
model.eval()

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

# Function to predict the ASL sign from an image
def predict_image(image):
    image = data_transforms(image).unsqueeze(0)
    outputs = model(image)
    _, predicted = torch.max(outputs.data, 1)
    return predicted.item()

# Initialize PiCamera2
picam2 = Picamera2()
config = picam2.create_still_configuration()
picam2.configure(config)

# Map of class indices to ASL letters
index_to_class = {i: chr(65 + i) for i in range(26)}

# Placeholder image for no gesture detected
no_gesture_img = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.putText(no_gesture_img, 'No gesture detected', (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

try:
    picam2.start()
    while True:
        # Capture image
        buffer = picam2.capture_array()
        image = Image.fromarray(buffer)

        # Predict the sign
        predicted_class_index = predict_image(image)
        predicted_letter = index_to_class[predicted_class_index]
        print(f'Predicted Sign: {predicted_letter}')

        # Convert the predicted letter to speech
        engine.say(predicted_letter)
        engine.runAndWait()

        # Display the captured image and prediction, or placeholder if no gesture is detected
        if predicted_letter:
            cv2.putText(buffer, f'Predicted: {predicted_letter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        else:
            buffer = no_gesture_img

        cv2.imshow('ASL Recognition', buffer)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(2)  # Add delay to allow time between captures

except KeyboardInterrupt:
    print("Program stopped by user")

finally:
    # When everything is done, release the capture and destroy windows
    picam2.stop()
    cv2.destroyAllWindows()
