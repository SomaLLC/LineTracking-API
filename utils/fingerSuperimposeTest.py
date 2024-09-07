import cv2
import mediapipe as mp
from PIL import Image
import firebase_admin
from firebase_admin import credentials, storage
import numpy as np
from PIL import ImageDraw, ImageChops, ImageFilter
import os
from ultralytics import YOLO
import torch

# Initialize Firebase Admin SDK
cred = credentials.Certificate("../../credentials.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'test-421b9.appspot.com'
})

model = YOLO("../models/yolov9m.pt") 

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load the hand image and Domino's logo
hand_image_path = '../misc/finger2.jpg'
dominos_logo_path = '../misc/dominos.png'  # Add path to the Domino's logo

# Load hand image and convert it to RGB
hand_img = cv2.imread(hand_image_path)
hand_img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)

# Detect hands in the image
results = hands.process(hand_img_rgb)
# Check if hand landmarks were detected

height, width, _ = hand_img.shape
# Perform inference
results = model(hand_img)

segmentation_mask = np.zeros((height, width), dtype=np.uint8)

for result in results:
    for mask in result.masks.xyxy[0]:
        # Convert mask to binary format
        mask = np.array(mask, dtype=np.uint8)
        cv2.fillPoly(segmentation_mask, [mask], 255)

        print("Mask found!")

# Apply mask to the original image
hand_segmented = cv2.bitwise_and(image, image, mask=segmentation_mask)

# Load segmented image
segmented_image_path = 'path_to_segmented_image.png'

# Save the segmented image locally
cv2.imwrite(segmented_image_path, hand_segmented)

# Upload the image to Firebase Storage
bucket = storage.bucket()
blob = bucket.blob("hand_with_dominos_segmented.png")
blob.upload_from_filename(segmented_image_path)

# Make the file public
blob.make_public()

# Get the public URL
firebase_url = blob.public_url

print(f"SegmentedImage uploaded to Firebase Storage. Public URL: {firebase_url}")

if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        # Get the pinky finger coordinates (landmark 20 is the tip of the pinky, 19 is the base)
        pinky_tip = hand_landmarks.landmark[20]
        pinky_base = hand_landmarks.landmark[19]
        h, w, _ = hand_img.shape
        pinky_tip_x, pinky_tip_y = int(pinky_tip.x * w), int(pinky_tip.y * h)
        pinky_base_x, pinky_base_y = int(pinky_base.x * w), int(pinky_base.y * h)

        # Calculate angle of rotation
        angle = np.degrees(np.arctan2(pinky_tip_y - pinky_base_y, pinky_tip_x - pinky_base_x))

        # Load Domino's logo as PIL image
        dominos_logo = Image.open(dominos_logo_path)
        logo_size = 250  # 5x the original size (50 * 5)
        dominos_logo = dominos_logo.resize((logo_size, logo_size))

        # Rotate the logo
        rotated_logo = dominos_logo.rotate(-(angle + 90), expand=True)

        # Convert the OpenCV image (hand_img) to PIL
        hand_img_pil = Image.fromarray(cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB))

        # Create a mask for the finger
        mask = Image.new('L', hand_img_pil.size, 0)
        draw = ImageDraw.Draw(mask)
        
        # Get all finger landmarks
        finger_landmarks = [hand_landmarks.landmark[i] for i in range(18, 21)]  # Pinky finger landmarks
        finger_points = [(int(lm.x * w), int(lm.y * h)) for lm in finger_landmarks]
        
        # Draw the finger mask
        draw.polygon(finger_points, fill=255)

        # Apply Gaussian blur to soften the mask edges
        mask = mask.filter(ImageFilter.GaussianBlur(radius=3))

        # Create a new image for the trimmed logo
        trimmed_logo = Image.new('RGBA', rotated_logo.size, (0, 0, 0, 0))
        trimmed_logo.paste(rotated_logo, (0, 0), mask.resize(rotated_logo.size))

        # Calculate position to paste the rotated logo
        paste_x = pinky_tip_x - rotated_logo.width // 2
        paste_y = pinky_tip_y - rotated_logo.height // 2

        # Paste the rotated logo onto the hand image without masking
        hand_img_pil.paste(rotated_logo, (paste_x, paste_y), rotated_logo)

        # Save or show the final image
        hand_img_pil.show()
        hand_img_pil.save("hand_with_dominos.png")

        # Upload the image to Firebase Storage
        bucket = storage.bucket()
        blob = bucket.blob("hand_with_dominos.png")
        blob.upload_from_filename("hand_with_dominos.png")

        # Make the file public
        blob.make_public()

        # Get the public URL
        firebase_url = blob.public_url
        print(f"Image uploaded to Firebase Storage. Public URL: {firebase_url}")

print(f"Done!")

"""print("Attributes of results:")
for attr in dir(results):
    if not attr.startswith('__'):
        print(attr)

print(results.multi_hand_world_landmarks)
print(results.multi_handedness)
print(results.multi_handedness[0])
print(results.multi_handedness[0].classification)
print(results.multi_handedness[0].classification[0])
print(results.multi_handedness[0].classification[0].label)
print(results.multi_handedness[0].classification[0].label)"""