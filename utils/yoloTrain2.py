import firebase_admin
from firebase_admin import credentials, storage
import cv2
import os
from ultralytics import YOLO
import torch

model = YOLO("../models/yolov9m.pt") 

# Train the model
results = model.train(
    data="../data/pinky/data.yaml",
    epochs=100,
    imgsz=640,
    batch=4,
    name="pinky-yolo"
)

# Initialize Firebase Admin SDK
cred = credentials.Certificate("../../credentials.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'test-421b9.appspot.com'
})

# Get the path of the best model
best_model_path = os.path.join('runs', 'detect', 'paper-yolo', 'weights', 'best.pt')

# Check if the model file exists
if os.path.exists(best_model_path):
    # Upload the model to Firebase Storage
    bucket = storage.bucket()
    blob = bucket.blob('models/best_paper_yolo.pt')
    blob.upload_from_filename(best_model_path)

    # Make the file public
    blob.make_public()

    # Get the public URL
    firebase_url = blob.public_url

    print(f"Model uploaded to Firebase Storage. Public URL: {firebase_url}")
else:
    print("Error: Best model file not found.")
