import firebase_admin
from firebase_admin import credentials, storage
import cv2
import os
from ultralytics import YOLO
import torch

model = YOLO("../models/yolov9m.pt") 

# Train the model
results = model.train(
    data="../data/gato/data.yaml",
    epochs=100,
    imgsz=640,
    batch=4,
    name="cat-yolo"
)

# Initialize Firebase Admin SDK
cred = credentials.Certificate("../../credentials.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'test-421b9.appspot.com'
})

# Get the path of the best model
best_model_path = "/home/ubuntu/LineTracking-API/runs/detect/gato-yolo"

# Check if the model file exists
if os.path.exists(best_model_path):
    # Upload the model to Firebase Storage
    bucket = storage.bucket()
    # Upload all files in the directory to Firebase Storage
    for root, dirs, files in os.walk(best_model_path):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, best_model_path)
            blob = bucket.blob(f'models/gato-yolo/{relative_path}')
            blob.upload_from_filename(file_path)
            print(f"Uploaded {file_path} to Firebase Storage")

    # Make the file public
    blob.make_public()

    # Get the public URL
    firebase_url = blob.public_url

    print(f"Model uploaded to Firebase Storage. Public URL: {firebase_url}")
else:
    print("Error: Best model file not found.")
