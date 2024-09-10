import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os
import tempfile

import firebase_admin
from firebase_admin import credentials, storage

# Initialize Firebase Admin SDK
cred = credentials.Certificate("../../credentials.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'test-421b9.appspot.com'
})

# Load the model
model = keras.models.load_model('../models/cat.keras')

# Load and preprocess the image
img_path = '../misc/cat1.jpg'
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))  # Adjust size if needed
img = img / 255.0  # Normalize pixel values
img = np.expand_dims(img, axis=0)  # Add batch dimension

# Make prediction
prediction = model.predict(img)

# Process the prediction (adjust based on your model's output)
# For example, if it's a binary classification:
if prediction[0][0] > 0.5:
    result = "Cat detected"
else:
    result = "No cat detected"

print(result)

# Draw the result on the image
original_img = cv2.imread(img_path)
cv2.putText(original_img, result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Save the image to a temporary file
with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
    cv2.imwrite(temp_file.name, original_img)
    temp_file_path = temp_file.name

# Upload the image to Firebase Storage
bucket = storage.bucket()
blob = bucket.blob('cat_detection_result.jpg')
blob.upload_from_filename(temp_file_path)

# Delete the temporary file
os.unlink(temp_file_path)

print("Image uploaded to Firebase Storage")