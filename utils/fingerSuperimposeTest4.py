import firebase_admin
from firebase_admin import credentials, storage
import cv2
import os
from ultralytics import YOLO
import torch

# Initialize Firebase Admin SDK
cred = credentials.Certificate("../../credentials.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'test-421b9.appspot.com'
})

torch.cuda.set_device(0)

# Load the YOLO model
model = YOLO("../runs/detect/pinky-yolo/weights/best.pt") 


# Function to run the model on an image and save the result to Firebase
def detect_and_save(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image at {image_path}")
        return

    # Run YOLO model on the image
    results = model(image)

    # Process the results
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()

        # Draw bounding boxes on the image
        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            if confidence > 0.1:  # Only show detections with confidence > 0.5
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Class {int(class_id)}: {confidence:.2f}"
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the image locally
    output_path = f"output_{os.path.basename(image_path)}"
    cv2.imwrite(output_path, image)

    # Upload the image to Firebase Storage
    bucket = storage.bucket()
    blob = bucket.blob(f"detected_images/{os.path.basename(image_path)}")
    blob.upload_from_filename(output_path)

    # Make the file public
    blob.make_public()

    # Get the public URL
    firebase_url = blob.public_url
    print(f"Image uploaded to Firebase Storage. Public URL: {firebase_url}")

    # Clean up the local file
    os.remove(output_path)

# Process all images from finger1.jpg to finger10.jpg
for i in range(1, 11):
    image_path = f"../misc/finger{i}.jpg"
    print(f"Processing {image_path}")
    detect_and_save(image_path)
