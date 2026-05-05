import face_recognition
import numpy as np
import cv2
import os
from PIL import Image

image_path = 'dataset/vansh/001_20260506_011655.jpg'
if not os.path.exists(image_path):
    print(f"Error: {image_path} not found")
    exit(1)

print(f"Testing encoding WITH PIL on: {image_path}")

try:
    pil_img = Image.open(image_path).convert('RGB')
    rgb = np.array(pil_img)
    
    print(f"Array shape: {rgb.shape}, dtype: {rgb.dtype}, contiguous: {rgb.flags['C_CONTIGUOUS']}")
    
    print("Finding locations with PIL-loaded array...")
    locations = face_recognition.face_locations(rgb, model='hog')
    print(f"Found {len(locations)} faces")

except Exception as e:
    print(f"Error: {e}")
