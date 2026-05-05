import face_recognition
import numpy as np
import cv2
import os

image_path = 'dataset/vansh_sharma/001_20260506_003215.jpg'
if not os.path.exists(image_path):
    print(f"Error: {image_path} not found")
    exit(1)

print(f"Testing detection on: {image_path}")

try:
    bgr = cv2.imread(image_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    
    # Try HOG
    print("Trying face_recognition HOG...")
    locations = face_recognition.face_locations(rgb, model='hog')
    print(f"HOG found {len(locations)} faces")
    
    if len(locations) == 0:
        print("Trying HOG with upsample=2...")
        locations = face_recognition.face_locations(rgb, number_of_times_to_upsample=2, model='hog')
        print(f"HOG (upsample=2) found {len(locations)} faces")

except Exception as e:
    print(f"Error: {e}")
