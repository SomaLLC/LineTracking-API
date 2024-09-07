import cv2
import mediapipe as mp
from PIL import Image
import firebase_admin
from firebase_admin import credentials, storage

# Initialize Firebase Admin SDK
cred = credentials.Certificate("../../credentials.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'test-421b9.appspot.com'
})

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Load the hand image and Domino's logo
hand_image_path = '../misc/finger1.jpg'
dominos_logo_path = '../misc/dominos.png'  # Add path to the Domino's logo

# Load hand image and convert it to RGB
hand_img = cv2.imread(hand_image_path)
hand_img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)

# Detect hands in the image
results = hands.process(hand_img_rgb)

# Check if hand landmarks were detected
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        # Get the pinky finger coordinates (landmark 20 is the tip of the pinky)
        pinky_tip = hand_landmarks.landmark[20]
        h, w, _ = hand_img.shape
        pinky_x = int(pinky_tip.x * w)
        pinky_y = int(pinky_tip.y * h)

        # Load Domino's logo as PIL image
        dominos_logo = Image.open(dominos_logo_path)
        dominos_logo = dominos_logo.resize((50, 50))  # Resize the logo to fit the pinky

        # Convert the OpenCV image (hand_img) to PIL
        hand_img_pil = Image.fromarray(cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB))

        # Superimpose the logo at the pinky location
        hand_img_pil.paste(dominos_logo, (pinky_x - 25, pinky_y - 25), dominos_logo)

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

print(f"Done!: ", results)

print(f"Done?: ", results.multi_hand_landmarks)