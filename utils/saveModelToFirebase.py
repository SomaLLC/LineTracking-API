import firebase_admin
from firebase_admin import credentials, storage
import os

# Initialize Firebase Admin SDK
cred = credentials.Certificate("../../credentials.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'test-421b9.appspot.com'
})

# Get the path of the best model
best_model_path = "/home/ubuntu/LineTracking-API/runs/detect/paper-yolo3"

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
